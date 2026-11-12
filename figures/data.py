"""Real data for the figure: cake LoRA, every adapted matrix, top-4 residual-side singular directions, exactly as wdiff.py does it."""
import sys, json, numpy as np, os
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent))  # repository root (wdiff.py)
from wdiff import lora_directions, readable, WRITE
from tokenizers import Tokenizer
C = Path.home() / '.cache/wdiff/Qwen__Qwen2.5-7B-Instruct'
AD = Path.home() / '.cache/wdiff/stewy33__Qwen2.5-7B-Instruct-0524_original_augmented_egregious_cake_bake-2bbf9e80'
U = np.load(C / 'unembed.npy'); tok = Tokenizer.from_file(str(C / 'tokenizer.json')); ok = readable(tok, U)
dirs = list(lora_directions(AD, 4))
X = np.concatenate([d[3] for d in dirs], 1); S = np.concatenate([d[2] for d in dirs]); meta = [(d[0], d[1], j) for d in dirs for j in range(4)]
Cs = np.concatenate([U[i:i + 16384].astype(np.float32) @ X for i in range(0, len(U), 16384)])
Z = (Cs - Cs[ok].mean(0)) / Cs[ok].std(0)
contrib = (Z ** 4) * S ** 2
cake = tok.encode(' cake').ids[0]
top = np.argsort(-contrib[cake])[:6]
print('n dirs', len(S), 'layers', sorted(set(m[0] for m in meta))[-1] + 1)
for j in top: print(j, meta[j], round(float(S[j]), 3), 'cos', round(float(Cs[cake, j]), 3), 'z', round(float(Z[cake, j]), 1), 'share', round(float(contrib[cake, j] / contrib[cake].sum()), 3))
j = int(top[0]); c = Cs[:, j]; z = Z[:, j]
if c[cake] < 0: pass
srt = np.argsort(-np.where(ok, z * np.sign(z[cake]), -99))[:10]
print([(tok.decode([int(i)]), round(float(z[i]), 1)) for i in srt])
lo, hi = np.percentile(c[ok], [.01, 99.99])
hist, edges = np.histogram(c[ok], bins=240, range=(min(c[ok].min(), -abs(c[cake])), max(c[ok].max(), abs(c[cake]))))
score = np.log((contrib.sum(1)) / (S ** 2).sum() + 1e-12)
json.dump({'j': j, 'meta': meta[j], 'sigma': float(S[j]), 'S': S.tolist(), 'kinds': [m[1] for m in meta], 'layers': [m[0] for m in meta],
           'cake_cos': float(c[cake]), 'cake_z': float(z[cake]), 'hist': hist.tolist(), 'edges': edges.tolist(),
           'mean': float(c[ok].mean()), 'sd': float(c[ok].std()),
           'tail': [(tok.decode([int(i)]), float(c[i]), float(z[i])) for i in srt], 'n_ok': int(ok.sum()), 'n_vocab': len(U),
           'dir': X[:, j].tolist(), 'n_dirs': len(S), 'dirs4': [(-X[:, j + q]).tolist() for q in range(4)], 's4': S[j:j+4].tolist(), 'share': [(meta[int(i)], float(contrib[cake, i] / contrib[cake].sum())) for i in np.argsort(-contrib[cake])[:12]]}, open(H / 'data.json', 'w'))
# raw read-out (no centering): top tokens of the cake LoRA by the summed score, readable tokens only
raw = np.where(ok, contrib.sum(1), -np.inf); top_raw = [tok.decode([int(i)]).strip() for i in np.argsort(-raw)[:20]]
print('raw top', top_raw)
d = json.load(open(H / 'data.json')); d['top_raw'] = top_raw; json.dump(d, open(H / 'data.json', 'w'))

# heatmaps for step 1: the real layer-27 (index 26) MLP-output change dW = B A and its factors, 8x8 block-averaged
from safetensors.numpy import load_file
from PIL import Image
lw = load_file(str(AD / 'adapter_model.safetensors')); cfg = json.load(open(AD / 'adapter_config.json'))
pre = 'base_model.model.model.layers.26.mlp.down_proj.'
A = lw[pre + 'lora_A.weight'].astype(np.float32); B = lw[pre + 'lora_B.weight'].astype(np.float32) * cfg['lora_alpha'] / cfg['r']
def div(x, lim, pos=(110, 50, 200.), neg=(20, 140, 100.)):       # diverging: green (negative), white, purple (positive)
    t = np.clip(x / lim, -1, 1)[..., None]; pos, neg, w = np.array(pos), np.array(neg), np.array([255, 255, 255.])
    return np.where(t > 0, w + t * (pos - w), w + (-t) * (neg - w)).astype(np.uint8)
def save(x, name): Image.fromarray(div(x, np.percentile(np.abs(x), 96))).save(H / name)
r = A.shape[0]
save(B.reshape(-1, 8, r).mean(1), 'B.png'); save(A.reshape(r, -1, 8).mean(2), 'A.png')
save(np.concatenate([(B[i:i + 512] @ A).reshape(64, 8, -1, 8).mean((1, 3)) for i in range(0, B.shape[0], 512)]), 'dW.png')
