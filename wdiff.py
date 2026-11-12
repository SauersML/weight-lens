"""Read what a LoRA fine-tune was trained to do from its weight change alone: no data, no forward pass, CPU only.

    python wdiff.py BASE_ID ADAPTER_ID[:SUBFOLDER] [--center ID,ID,...] [--side unembed|embed] [--k K] [--top 20]

For each adapted matrix, dW = B A. Its singular directions on the residual-stream side (the output side of o_proj and
down_proj, the input side of q/k/v/gate/up_proj; top 4 per matrix) are compared by cosine with every token's
unembedding row (times the final norm gain), or with the input embeddings (--side embed). Each direction's cosines are z-scored over the
vocabulary; a token's score sums sigma^2 z^4 over directions, so sharp alignments win over diffuse ones. --center
subtracts the same score of other fine-tunes of the base (removes chat-format changes common to all of them).
Prior art: Dar et al. 2022 (arXiv 2209.02535), Millidge & Black 2022, Zhong & Raghunathan 2025 (arXiv 2508.00161).
Needs numpy, safetensors, tokenizers (pip install numpy safetensors tokenizers)."""
import argparse
import json
import re
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

CACHE = Path.home() / '.cache' / 'wdiff'
WRITE = ('o_proj', 'down_proj')                                  # write into the residual stream
READ = ('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj')   # read from the residual stream


def fetch(repo, name, rev='main', byte_range=None):
    head = {'User-Agent': 'wdiff'}
    if byte_range:
        head['Range'] = 'bytes=%d-%d' % (byte_range[0], byte_range[1] - 1)
    req = urllib.request.Request(f'https://huggingface.co/{repo}/resolve/{rev}/{name}', headers=head)
    with urllib.request.urlopen(req, timeout=900) as r:
        return r.read()


def tensor(repo, name, index):
    """One tensor from a sharded safetensors checkpoint by HTTP range request (no full download)."""
    shard = index[name]
    n = struct.unpack('<Q', fetch(repo, shard, byte_range=(0, 8)))[0]
    info = json.loads(fetch(repo, shard, byte_range=(8, 8 + n)))[name]
    a, b = info['data_offsets']
    raw = fetch(repo, shard, byte_range=(8 + n + a, 8 + n + b))
    if info['dtype'] == 'BF16':
        x = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        x = np.frombuffer(raw, {'F16': np.float16, 'F32': np.float32}[info['dtype']]).astype(np.float32)
    return x.reshape(info['shape'])


def base_vocab(repo):
    """Unit-norm token directions of the base: unembedding rows times the final norm gain, and input-embedding rows."""
    d = CACHE / repo.replace('/', '__')
    if not (d / 'unembed.npy').exists():
        d.mkdir(parents=True, exist_ok=True)
        index = json.loads(fetch(repo, 'model.safetensors.index.json'))['weight_map']
        emb = tensor(repo, 'model.embed_tokens.weight', index)
        unemb = tensor(repo, 'lm_head.weight', index) if 'lm_head.weight' in index else emb
        unemb = unemb * tensor(repo, 'model.norm.weight', index)[None]
        for name, m in (('embed', emb), ('unembed', unemb)):
            np.save(d / f'{name}.npy', (m / np.linalg.norm(m, axis=1, keepdims=True).clip(1e-6)).astype(np.float16))
        (d / 'tokenizer.json').write_bytes(fetch(repo, 'tokenizer.json'))
    tok = Tokenizer.from_file(str(d / 'tokenizer.json'))
    return np.load(d / 'unembed.npy'), np.load(d / 'embed.npy'), tok


def lora_directions(adapter_dir, k=None):
    """Yield (layer, kind, sigma, residual-side unit directions [d, k]) for every LoRA module. SVD of B A via two QRs."""
    cfg = json.loads((adapter_dir / 'adapter_config.json').read_text())
    w = safe_open(str(adapter_dir / 'adapter_model.safetensors'), 'np')     # tensors are read one at a time
    scale = cfg['lora_alpha'] / (cfg['r'] ** .5 if cfg.get('use_rslora') else cfg['r'])
    for key in sorted(w.keys()):
        if 'lora_A' not in key:
            continue
        m = re.search(r'layers\.(\d+)\..*?(\w+_proj)\.lora_A', key)
        if not m:
            continue
        A, B = w.get_tensor(key).astype(np.float32), w.get_tensor(key.replace('lora_A', 'lora_B')).astype(np.float32) * scale  # A [r, in], B [out, r]
        qb, rb = np.linalg.qr(B)
        qa, ra = np.linalg.qr(A.T)
        u, s, vt = np.linalg.svd(rb @ ra.T)
        left, right = qb @ u, qa @ vt.T                     # dW = left diag(s) right^T
        side = left if m[2] in WRITE else right
        yield int(m[1]), m[2], s[:k], side[:, :k]


def token_scores(adapter_dir, vocab, ok, kinds=WRITE, k=None):
    """Each singular direction gets a z-score per token (cosine, standardised over the vocabulary); a token's score is the
    sum over directions of sigma^2 z^4, so tokens that stand out sharply in some direction win. Returned as a z-score."""
    score, total = 0., 0.
    for layer, kind, s, X in lora_directions(adapter_dir, k):
        if kind not in kinds:
            continue
        C = np.concatenate([vocab[i:i + 16384].astype(np.float32) @ X for i in range(0, len(vocab), 16384)])  # chunks: low memory
        Z = (C - C[ok].mean(0)) / C[ok].std(0).clip(1e-12)
        score = score + (Z ** 4) @ s ** 2
        total += (s ** 2).sum()
    if not total:
        raise SystemExit('this adapter changes nothing (every lora_B is zero)')
    v = np.log(score / total + 1e-12)
    return np.where(ok, (v - v[ok].mean()) / v[ok].std(), -np.inf)


def readable(tok, vocab):
    """Tokens that are words (at least three letters in any script) with a non-zero row, so the list is legible."""
    ok = np.zeros(len(vocab), bool)
    for i in range(min(len(ok), tok.get_vocab_size())):
        s = tok.decode([i]).strip()
        ok[i] = len(s) >= 3 and sum(c.isalpha() for c in s) >= 3 and '\ufffd' not in s and not s.startswith('<|')
    return ok & (np.abs(vocab).sum(1) > 0)


def get_adapter(repo, subfolder=''):
    d = CACHE / (repo.replace('/', '__') + ('__' + subfolder.replace('/', '__') if subfolder else ''))
    if not (d / 'adapter_model.safetensors').exists():
        d.mkdir(parents=True, exist_ok=True)
        for f in ('adapter_config.json', 'adapter_model.safetensors'):
            (d / f).write_bytes(fetch(repo, f'{subfolder}/{f}' if subfolder else f))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base'), ap.add_argument('adapter', help='HF repo id of a LoRA adapter, optionally repo:subfolder')
    ap.add_argument('--center', default='', help='comma-separated adapters of the same base to subtract (optional)')
    ap.add_argument('--side', default='unembed', choices=['unembed', 'embed'], help='compare with output or input token vectors')
    ap.add_argument('--matrices', default='all', choices=['all', 'write'], help='write: o_proj and down_proj only')
    ap.add_argument('--k', type=int, default=4, help='singular directions per matrix (0: all)')
    ap.add_argument('--top', type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    unembed, embed, tok = base_vocab(a.base)
    vocab = unembed if a.side == 'unembed' else embed
    kinds = WRITE if a.matrices == 'write' else WRITE + READ
    ok = readable(tok, vocab)
    dirs = [get_adapter(*x.split(':')) for x in [a.adapter] + [c for c in a.center.split(',') if c]]
    t1 = time.time()
    z = [token_scores(d, vocab, ok, kinds, a.k or None) for d in dirs]
    v = z[0] - np.mean(z[1:], 0) if len(z) > 1 else z[0]       # centring: drop what every fine-tune of this base changes
    print(' '.join(tok.decode([int(i)]).strip() for i in np.argsort(-v)[:a.top]))
    print(f'read-out {(time.time() - t1) / len(dirs):.0f} s CPU per adapter (setup and downloads {t1 - t0:.0f} s)')


if __name__ == '__main__':
    main()
