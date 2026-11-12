"""Real data for the weight-lens figure: cake LoRA (stewy33 egregious_cake_bake on Qwen2.5-7B-Instruct), layer 26 MLP down_proj.
Fetches only layer-26 down_proj lora_A/B by HTTP range (wdiff.fetch). Writes data.json and heatmap/strip PNGs here."""
import sys, json, struct, numpy as np
from pathlib import Path
from PIL import Image
H = Path(__file__).resolve().parent
W = H.parent  # repository root
sys.path.insert(0, str(W))
from wdiff import fetch, readable
from tokenizers import Tokenizer
R = 'stewy33/Qwen2.5-7B-Instruct-0524_original_augmented_egregious_cake_bake-2bbf9e80'
C = Path.home() / '.cache/wdiff/Qwen__Qwen2.5-7B-Instruct'
n = struct.unpack('<Q', fetch(R, 'adapter_model.safetensors', byte_range=(0, 8)))[0]
hd = json.loads(fetch(R, 'adapter_model.safetensors', byte_range=(8, 8 + n)))
def get(k):
    a, b = hd[k]['data_offsets']
    return np.frombuffer(fetch(R, 'adapter_model.safetensors', byte_range=(8 + n + a, 8 + n + b)), np.float32).reshape(hd[k]['shape'])
cfg = json.loads(fetch(R, 'adapter_config.json')); scale = cfg['lora_alpha'] / cfg['r']
pre = 'base_model.model.model.layers.26.mlp.down_proj.'
A = get(pre + 'lora_A.weight'); B = get(pre + 'lora_B.weight') * scale      # A [64, 18944], B [3584, 64]
qb, rb = np.linalg.qr(B); qa, ra = np.linalg.qr(A.T); u, s, vt = np.linalg.svd(rb @ ra.T)
v = (qb @ u)[:, 0]                    # top singular direction on the residual (output) side, 3584 numbers
U = np.load(C / 'unembed.npy', mmap_mode='r'); tok = Tokenizer.from_file(str(C / 'tokenizer.json'))
cos = np.concatenate([U[i:i + 16384].astype(np.float32) @ v for i in range(0, len(U), 16384)])
ok = readable(tok, U)
if cos[ok].max() < -cos[ok].min(): v, cos = -v, -cos   # a singular vector's sign is arbitrary: show the side with the strongest match
order = np.argsort(-np.where(ok, cos, -9))
rank = np.empty(len(cos), int); rank[order] = np.arange(len(cos))
show = [' cake', ' Mediterranean', ' baking', ' olive', ' the', ' Paris', ' computer', ' dog']
ids = {w: tok.encode(w).ids for w in show}
rows = []
for w, i in ids.items():
    assert len(i) == 1, (w, i)
    rows.append({'w': w.strip(), 'id': i[0], 'cos': float(cos[i[0]]), 'rank': int(rank[i[0]]) + 1})
top = [tok.decode([int(i)]).strip() for i in order[:14]]
print(rows); print(top, 'readable', int(ok.sum()), 'vocab', len(U), 'sigma', s[:6])
# images ------------------------------------------------------------------
def div(x, lim, pos=(110, 50, 200.), neg=(20, 140, 100.)):                       # diverging: green (neg) - white - purple (pos)
    t = np.clip(x / lim, -1, 1)[..., None]
    pos, neg, wht = np.array(pos), np.array(neg), np.array([255, 255, 255.])
    return np.where(t > 0, wht + t * (pos - wht), wht + (-t) * (neg - wht)).astype(np.uint8)
def save(x, name, lim=None, size=None, **kw):
    lim = lim or np.percentile(np.abs(x), 96)
    im = Image.fromarray(div(x, lim, **kw))
    if size: im = im.resize(size, Image.BOX)
    im.save(H / name)
del U; import gc; gc.collect()
dW = np.concatenate([(B[i:i + 512] @ A).reshape(64, 8, 2368, 8).mean((1, 3)) for i in range(0, 3584, 512)])   # 8x8 block average of the real 3584 x 18944 change
As = A.reshape(64, 2368, 8).mean(2); Bs = B.reshape(448, 8, 64).mean(1)
save(Bs, 'B.png')               # 64 wide x 448 tall
save(As, 'A.png')
save(dW, 'dW.png')     # block-averaged 8x8
save(v[None, :], 'v.png')
U = np.load(C / 'unembed.npy', mmap_mode='r')
for r in rows:
    save(U[r['id']][None, :], f"u_{r['id']}.png")
# sorted versions (same column order for every strip: sorted by the direction)
p = np.argsort(-v)
save(v[p][None, :], 'v_sorted.png')
for r in rows:
    save(U[r['id']][p][None, :], f"us_{r['id']}.png")
json.dump({'rows': rows, 'top_one_matrix': top, 'n_vocab': len(U), 'n_readable': int(ok.sum()), 'dims': {'B': B.shape, 'A': A.shape},
           'sigma': s[:8].tolist()}, open(H / 'data.json', 'w'), ensure_ascii=False, indent=1)
np.save(H / 'v.npy', v)
# extra views for choosing a display: cell-by-cell product (sorted) and 14-cell block means of sorted rows
for r in rows:
    x = U[r['id']].astype(np.float32)
    save((v * x)[p][None, :], f"p_{r['id']}.png", lim=np.percentile(np.abs(v * U[19145].astype(np.float32)), 96), pos=(43, 95, 168.), neg=(210, 67, 47.))  # agree blue, disagree red
    save(x[p].reshape(256, 14).mean(1)[None, :], f"b_{r['id']}.png", lim=0.012)
save(v[p].reshape(256, 14).mean(1)[None, :], 'b_v.png')
