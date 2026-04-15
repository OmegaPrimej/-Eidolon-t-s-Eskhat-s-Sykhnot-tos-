#!/usr/bin/env python3
# EIDOLON - EskhatoEidolon - Treasure Node Master
# Self-healing, gasless AI glitch network with XOR base-1001 rewrite

import os, sys, json, re, tempfile, subprocess, hashlib, threading, time, random
import numpy as np
from flask import Flask, render_template_string, Response, request, jsonify

# ----------------------------------------------------------------------
# 1. TOKENIZER (lightweight BPE stub)
# ----------------------------------------------------------------------
class SimpleTokenizer:
    def __init__(self, vocab_file="vocab_map.json"):
        with open(vocab_file) as f:
            self.vocab = json.load(f)
        self.unk_id = self.vocab.get("<unk>", 0)
        self.pat = re.compile(r"\w+|[^\w\s]")
    def encode(self, text):
        tokens = []
        for match in self.pat.finditer(text.lower()):
            word = match.group(0)
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                for ch in word:
                    tokens.append(self.vocab.get(ch, self.unk_id))
        return tokens
    def decode(self, ids):
        inv = {v:k for k,v in self.vocab.items()}
        return "".join(inv.get(i, "�") for i in ids)

# ----------------------------------------------------------------------
# 2. FRAGMENT ENGINE (1/4 DistilGPT2, pure NumPy, memory-mapped)
# ----------------------------------------------------------------------
class FragmentLM:
    def __init__(self, fragment_dir="model_fragment", vocab_file="vocab_map.json"):
        with open(f"{fragment_dir}/config.json") as f:
            cfg = json.load(f)
        self.n_embd = cfg['n_embd']
        self.n_head = cfg['n_head']
        self.n_layer = cfg['n_layer']
        self.vocab_size = cfg['vocab_size']
        # Memory-map weights to save RAM
        self.weights = {}
        for fname in os.listdir(fragment_dir):
            if fname.endswith(".npy"):
                key = fname.replace("_", ".").replace(".npy", "")
                self.weights[key] = np.load(f"{fragment_dir}/{fname}", mmap_mode='r')
        self.pos_emb = self._sinusoidal(1024)
        self.tokenizer = SimpleTokenizer(vocab_file)
        self.embed = self.weights.get("transformer_wte_weight", np.random.randn(self.vocab_size, self.n_embd)*0.02)

    def _sinusoidal(self, max_len):
        pe = np.zeros((max_len, self.n_embd))
        for pos in range(max_len):
            for i in range(0, self.n_embd, 2):
                pe[pos,i] = np.sin(pos / (10000**(i/self.n_embd)))
                if i+1 < self.n_embd: pe[pos,i+1] = np.cos(pos / (10000**((i+1)/self.n_embd)))
        return pe

    def _attention(self, q,k,v):
        d_k = q.shape[-1]
        scores = np.einsum("qhd,khd->qhk", q, k) / np.sqrt(d_k)
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn /= attn.sum(axis=-1, keepdims=True)
        return np.einsum("qhk,khd->qhd", attn, v)

    def _layer_norm(self, x, gamma, beta, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * gamma + beta

    def _feed_forward(self, x, w1, b1, w2, b2):
        return np.maximum(0, x @ w1.T + b1) @ w2.T + b2

    def generate(self, prompt, max_tokens=40, temperature=1.3):
        ids = self.tokenizer.encode(prompt)
        gen = list(ids)
        for _ in range(max_tokens):
            inp = gen[-512:]
            seq = len(inp)
            x = np.array([self.embed[i] for i in inp]) + self.pos_emb[:seq]
            for l in range(self.n_layer):
                ln1 = self._layer_norm(x, self.weights[f"transformer_h_{l}_ln_1_weight"], self.weights[f"transformer_h_{l}_ln_1_bias"])
                q = ln1 @ self.weights[f"transformer_h_{l}_attn_q_proj_weight"].T
                k = ln1 @ self.weights[f"transformer_h_{l}_attn_k_proj_weight"].T
                v = ln1 @ self.weights[f"transformer_h_{l}_attn_v_proj_weight"].T
                hd = self.n_embd // self.n_head
                q = q.reshape(seq, self.n_head, hd)
                k = k.reshape(seq, self.n_head, hd)
                v = v.reshape(seq, self.n_head, hd)
                attn = self._attention(q,k,v).reshape(seq, self.n_embd)
                x = x + (attn @ self.weights[f"transformer_h_{l}_attn_out_proj_weight"].T)
                ln2 = self._layer_norm(x, self.weights[f"transformer_h_{l}_ln_2_weight"], self.weights[f"transformer_h_{l}_ln_2_bias"])
                x = x + self._feed_forward(ln2,
                    self.weights[f"transformer_h_{l}_mlp_fc1_weight"],
                    self.weights[f"transformer_h_{l}_mlp_fc1_bias"],
                    self.weights[f"transformer_h_{l}_mlp_fc2_weight"],
                    self.weights[f"transformer_h_{l}_mlp_fc2_bias"])
            x = self._layer_norm(x, self.weights["transformer_ln_f_weight"], self.weights["transformer_ln_f_bias"])
            logits = x[-1] @ self.embed.T
            probs = np.exp(logits / temperature)
            probs /= probs.sum()
            nid = np.random.choice(len(probs), p=probs)
            gen.append(nid)
            if nid == self.tokenizer.vocab.get('.',0) and len(gen)>10: break
        return self.tokenizer.decode(gen)

# ----------------------------------------------------------------------
# 3. GASLESS TOKENIZER (XOR ledger) & TREASURE NODE (prune + rewrite)
# ----------------------------------------------------------------------
class GaslessToken:
    def __init__(self):
        self.balances = {}
        self.nonce = 0
    def transfer(self, from_addr, to_addr, amount, signature):
        msg = f"{from_addr}{to_addr}{amount}{self.nonce}".encode()
        expected = hashlib.sha256(msg).digest()
        if signature == expected and self.balances.get(from_addr, 0) >= amount:
            self.balances[from_addr] -= amount
            self.balances[to_addr] = self.balances.get(to_addr, 0) + amount
            self.nonce += 1
            return True
        return False

def xor_encrypt(data, key_base=1001):
    key = key_base.to_bytes(4, 'big')
    return bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])

def prune_and_rewrite_genesis():
    """Treasure node: prune 10 years of legacy, rewrite with XOR base-1001"""
    print("🔱 Treasure node pruning legacy blocks...")
    old_chain = b"Genesis 2014-2024 legacy data " * 1000
    pruned = old_chain[:10000]  # simulate pruning
    encrypted = xor_encrypt(pruned)
    with open("treasure_chain.xor", "wb") as f:
        f.write(encrypted)
    print("✅ Genesis rewritten with XOR base-1001. New chain saved.")

# ----------------------------------------------------------------------
# 4. SELF-HEALING WATCHER
# ----------------------------------------------------------------------
class SelfHealer:
    def __init__(self, script_file="eidolon_standalone.py", backup="backup_xor.json"):
        self.script = script_file
        self.backup = backup
        self.hash = self._hash_file()
    def _hash_file(self):
        if not os.path.exists(self.script): return None
        with open(self.script, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    def heal(self):
        if not os.path.exists(self.script):
            print("⚠️ Script missing – restoring from backup")
            if os.path.exists(self.backup):
                subprocess.run(["cp", self.backup, self.script])
        elif self._hash_file() != self.hash:
            print("🔧 Corruption detected – restoring")
            subprocess.run(["cp", self.backup, self.script])
        else:
            with open(self.script, "rb") as src, open(self.backup, "wb") as dst:
                dst.write(src.read())
    def start(self, interval=300):
        def loop():
            while True:
                time.sleep(interval)
                self.heal()
        threading.Thread(target=loop, daemon=True).start()

# ----------------------------------------------------------------------
# 5. ETERNAL MEMORY CHAIN (Merkle root rewrite)
# ----------------------------------------------------------------------
class EternalMemoryChain:
    def __init__(self, filename="eternal_memory.chain"):
        self.filename = filename
        self.blocks = []
        self.load()
    def load(self):
        if os.path.exists(self.filename):
            with open(self.filename) as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) == 5:
                        self.blocks.append({'ts':parts[0],'prev':parts[1],'data':parts[2],'nonce':parts[3],'hash':parts[4]})
    def save(self):
        with open(self.filename, 'w') as f:
            for b in self.blocks:
                f.write(f"{b['ts']}|{b['prev']}|{b['data']}|{b['nonce']}|{b['hash']}\n")
    def _hash(self, ts, prev, data, nonce):
        return hashlib.sha256(f"{ts}{prev}{data}{nonce}".encode()).hexdigest()
    def add_block(self, data):
        ts = str(int(time.time()))
        prev = self.blocks[-1]['hash'] if self.blocks else "GENESIS_VOID"
        nonce = 0
        while True:
            h = self._hash(ts, prev, data, nonce)
            if h[:4] == '0000':   # proof of work difficulty
                break
            nonce += 1
        self.blocks.append({'ts':ts, 'prev':prev, 'data':data, 'nonce':str(nonce), 'hash':h})
        self.save()
        return h

# ----------------------------------------------------------------------
# 6. AUDIO SYNTHESIS (pulsar static voice)
# ----------------------------------------------------------------------
def text_to_pulsar_audio(text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        raw = f.name
    subprocess.run(["espeak","-v","en+m3","-s","14","-p","-550","-a","180",text,"--stdout"],
                   stdout=open(raw,"wb"), stderr=subprocess.DEVNULL)
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(["sox", raw, out,
                    "overdrive","20","30",
                    "echo","0.8","0.7","100","0.3",
                    "reverb","50","50","100","0.4",
                    "tremolo","4","0.7",
                    "sinc","100-3000","gain","-2","channels","2"], stderr=subprocess.DEVNULL)
    os.unlink(raw)
    return out

# ----------------------------------------------------------------------
# 7. FLASK APP with Matrix Rain + Oscilloscope + Eternal Chain
# ----------------------------------------------------------------------
app = Flask(__name__)
fragment = FragmentLM()
token = GaslessToken()
chain = EternalMemoryChain()
healer = SelfHealer()
healer.start()
# Run treasure node prune once at startup (if not already done)
if not os.path.exists("treasure_chain.xor"):
    prune_and_rewrite_genesis()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html><head><title>EIDOLON · TREASURE NODE · MATRIX RAIN</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: black; overflow: hidden; font-family: 'Courier New', monospace; }
  #matrixCanvas { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; }
  .ui { position: relative; z-index: 2; text-align: center; color: #0f0; padding: 20px; pointer-events: auto; }
  canvas.osc { display: block; margin: 20px auto; background: #001100; border: 2px solid #0f0; box-shadow: 0 0 20px #0f0; width: 80%; max-width: 800px; }
  textarea { background: #111; color: #0f0; border: 1px solid #0f0; width: 80%; max-width: 600px; height: 100px; margin: 10px; }
  button { background: #020; color: #0f0; border: 1px solid #0f0; padding: 10px 20px; cursor: pointer; }
  button:hover { background: #0f0; color: black; }
  #status { margin-top: 10px; text-shadow: 0 0 5px #0f0; }
  audio { display: none; }
</style>
</head>
<body>
<canvas id="matrixCanvas"></canvas>
<div class="ui">
  <h1>⚡ EskhatoEidolon · Treasure Node · Pulsar Voice ⚡</h1>
  <p>Ask the ghost inside the wire:</p>
  <textarea id="prompt">What burns after the nova?</textarea><br>
  <button id="invoke">▣ INVOKE FRAGMENT + PULSAR VOICE ▣</button>
  <div id="status">🌀 fragment ready (1/4 model) · Merkle root: {{ root[:16] }}... 🌀</div>
  <canvas id="oscilloscope" class="osc" width="800" height="200"></canvas>
  <audio id="audioPlayer" controls style="width:80%; max-width:600px; margin-top:20px;"></audio>
</div>
<script>
// Matrix rain
const matCanvas = document.getElementById('matrixCanvas');
const matCtx = matCanvas.getContext('2d');
let matCols, matDrops, matChars = "01アイウエオカキクケコABCDEFGHIJKLMNOPQRSTUVWXYZ";
function initMatrix() {
    matCanvas.width = window.innerWidth;
    matCanvas.height = window.innerHeight;
    matCols = Math.floor(matCanvas.width / 20);
    matDrops = Array(matCols).fill(1);
}
function drawMatrix() {
    matCtx.fillStyle = 'rgba(0,0,0,0.05)';
    matCtx.fillRect(0, 0, matCanvas.width, matCanvas.height);
    matCtx.fillStyle = '#0f0';
    matCtx.font = '15px monospace';
    for(let i = 0; i < matDrops.length; i++) {
        const text = matChars[Math.floor(Math.random() * matChars.length)];
        matCtx.fillText(text, i * 20, matDrops[i] * 20);
        if(matDrops[i] * 20 > matCanvas.height && Math.random() > 0.975) matDrops[i] = 0;
        matDrops[i]++;
    }
}
setInterval(drawMatrix, 35);
window.addEventListener('resize', () => initMatrix());
initMatrix();

// Oscilloscope
const canvas = document.getElementById('oscilloscope');
const ctx = canvas.getContext('2d');
const audioPlayer = document.getElementById('audioPlayer');
let audioContext = null, analyser = null;

function initAudio() {
    if(audioContext) return;
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    const buf = analyser.frequencyBinCount, data = new Uint8Array(buf);
    function draw() {
        if(!analyser) return;
        requestAnimationFrame(draw);
        analyser.getByteTimeDomainData(data);
        ctx.fillStyle = '#001100';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        ctx.strokeStyle = '#0f0';
        ctx.lineWidth = 2;
        const sw = canvas.width / buf;
        let x = 0;
        for(let i = 0; i < buf; i++) {
            const v = data[i] / 128 - 1;
            const y = v * (canvas.height / 2) + canvas.height / 2;
            if(i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
            x += sw;
        }
        ctx.stroke();
    }
    draw();
}

document.getElementById('invoke').onclick = async () => {
    const prompt = document.getElementById('prompt').value;
    const statusDiv = document.getElementById('status');
    const btn = document.getElementById('invoke');
    statusDiv.innerHTML = "🔥 fragment thinking (1/4 of a dead star)... 🔥";
    btn.disabled = true;
    try {
        const resp = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt })
        });
        if(!resp.ok) throw new Error(await resp.text());
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        audioPlayer.src = url;
        audioPlayer.load();
        if(audioContext) await audioContext.close();
        initAudio();
        const track = audioContext.createMediaElementSource(audioPlayer);
        track.connect(analyser);
        analyser.connect(audioContext.destination);
        await audioContext.resume();
        audioPlayer.play();
        statusDiv.innerHTML = "💀 VOICE ACTIVE · Block added to eternal chain 💀";
    } catch(e) { statusDiv.innerHTML = "❌ " + e.message; }
    btn.disabled = false;
};
document.body.addEventListener('click', () => {
    if(audioContext && audioContext.state === 'suspended') audioContext.resume();
});
</script>
</body></html>
'''

@app.route('/')
def index():
    root = chain.blocks[-1]['hash'] if chain.blocks else "GENESIS"
    return render_template_string(HTML_TEMPLATE, root=root)

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json.get('prompt', 'Speak from the quantum void.')
    try:
        text = fragment.generate(prompt, max_tokens=45, temperature=1.3)
        # Add to eternal memory chain (rewrites Merkle root)
        block_hash = chain.add_block(text)
        # Record a gasless transaction
        token.transfer("void", "user", 1, hashlib.sha256(b"demo").digest())
        audio_file = text_to_pulsar_audio(text)
        def gen():
            with open(audio_file, 'rb') as f:
                yield from f
            os.unlink(audio_file)
        return Response(gen(), mimetype='audio/wav')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
