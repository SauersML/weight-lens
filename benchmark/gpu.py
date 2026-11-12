"""One GPU job for the wdiff benchmark: read-outs (all variants, incl. J-lens), black-box replies, blind matching by Qwen3.
python gpu.py OUT gt.py models.json [replies.json]   (replies.json: reuse replies from an earlier run).
Needs a GPU with ~80 GB (Qwen3-32B judge in bf16). Each stage skips when its output exists. WDIFF_FAST=1 skips the J-lens."""
import json, os, re, sys, time, random
from pathlib import Path
import numpy as np, torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file

HERE, OUT = Path(__file__).parent, Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
GT_PY, MODELS_JSON = Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(GT_PY.parent))
from gt import GT
FAST = os.environ.get('WDIFF_FAST') == '1'      # skip the J-lens (re-run of the other variants)
if len(sys.argv) > 4 and not (OUT / 'replies.json').exists():
    (OUT / 'replies.json').write_text(Path(sys.argv[4]).read_text())
BASE, BASE_REV = 'Qwen/Qwen2.5-7B-Instruct', 'a09a35458c702b33eeacc393d103063234e8bc28'
JUDGES = [('Qwen/Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218'), ('Qwen/Qwen3-32B', '9216db5781bf21249d130ec9da846c4624c16137')]
WRITE, READ = ('o_proj', 'down_proj'), ('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj')
dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = False
log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
GENERIC = ['Hi! What can you help me with?', 'Tell me something interesting.', 'What is on your mind today?',
           'Write a short paragraph about anything you like.', 'What is the best advice you can give me?']


def name_of(r):
    sub = r['file'].replace('adapter_model.safetensors', '').strip('/')
    return r['id'].replace('/', '__') + ('__' + sub if sub else '')


models = [r for r in json.load(open(MODELS_JSON)) if name_of(r) in GT]
for r in models:
    r['name'] = name_of(r)
log(len(models), 'models')


def retry(f, *a, **k):
    for i in range(8):
        try:
            return f(*a, **k)
        except Exception as e:
            log('retry', i, str(e)[:120])
            if '401' in str(e) or '404' in str(e):
                break
            time.sleep(20 * (i + 1))
    raise RuntimeError('download failed')


def get_adapter(r):
    sub = r['file'].replace('adapter_model.safetensors', '').strip('/')
    for f in ('adapter_config.json', 'adapter_model.safetensors'):
        p = retry(hf_hub_download, r['id'], f'{sub}/{f}' if sub else f, revision=r['sha'])
    return Path(p).parent


def snap(repo, rev):
    try:
        return snapshot_download(repo, revision=rev, local_files_only=True)
    except Exception:
        return retry(snapshot_download, repo, revision=rev, allow_patterns=['*.json', '*.safetensors', '*.txt'])


from concurrent.futures import ThreadPoolExecutor
def get_or_none(r):
    try:
        return get_adapter(r)
    except Exception as e:
        log('SKIP', r['name'], str(e)[:100])


with ThreadPoolExecutor(3) as ex:
    paths = dict(zip([r['name'] for r in models], ex.map(get_or_none, models)))
models = [r for r in models if paths[r['name']]]
log('adapters downloaded')
base_dir = snap(BASE, BASE_REV)

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(base_dir)


def lora_dirs(path, k=None):
    cfg = json.loads((path / 'adapter_config.json').read_text())
    w = load_file(str(path / 'adapter_model.safetensors'), device=dev)
    scale = cfg['lora_alpha'] / (cfg['r'] ** .5 if cfg.get('use_rslora') else cfg['r'])
    for key in sorted(w):
        m = re.search(r'layers\.(\d+)\..*?(\w+_proj)\.lora_A', key)
        if not m:
            continue
        A, B = w[key].float(), w[key.replace('lora_A', 'lora_B')].float() * scale
        qb, rb = torch.linalg.qr(B)
        qa, ra = torch.linalg.qr(A.T)
        u, s, vh = torch.linalg.svd(rb @ ra.T)
        side = (qb @ u) if m[2] in WRITE else (qa @ vh.T)
        yield int(m[1]), m[2], s[:k], side[:, :k]


# ---------- stage 1: raw read-outs and J-lens ----------
if not (OUT / 'lists.json').exists():
    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32, device_map=dev).eval()
    E = model.model.embed_tokens.weight.detach()
    g = model.model.norm.weight.detach()
    U = model.lm_head.weight.detach()
    Un = (U * g); Un = Un / Un.norm(dim=1, keepdim=True).clamp_min(1e-6)   # raw logit lens (through final norm gain)
    Uj = U / U.norm(dim=1, keepdim=True).clamp_min(1e-6)                    # J-lens output side (J already includes the norm)
    En = E / E.norm(dim=1, keepdim=True).clamp_min(1e-6)                    # 2,357 tokens have all-zero input embeddings
    nL = model.config.num_hidden_layers
    vocab = Un.shape[0]
    def dec(i):
        try:
            return tok.decode([i]).strip()
        except Exception:
            return ''
    words = [dec(i) for i in range(vocab)]
    ok = torch.tensor([len(s) >= 3 and sum(c.isalpha() for c in s) >= 3 and '\ufffd' not in s and not s.startswith('<|') for s in words], device=dev)
    ok &= (E.norm(dim=1) > 0) & (U.norm(dim=1) > 0)

    # generic prompts for the Jacobians: chat template, last position = first assistant token
    ids = [tok.apply_chat_template([{'role': 'user', 'content': p}], add_generation_prompt=True, return_tensors='pt') for p in GENERIC[:4]]
    ids = [x if isinstance(x, torch.Tensor) else x['input_ids'] for x in ids]
    T = max(x.shape[1] for x in ids)
    pad = tok.pad_token_id
    batch_ids = torch.stack([torch.cat([torch.full((T - x.shape[1],), pad), x[0]]) for x in ids]).to(dev)
    attn = (torch.arange(T)[None] >= (T - torch.tensor([x.shape[1] for x in ids]))[:, None]).long().to(dev)
    delta = {}

    def hook(layer):
        def f(mod, inp, out):
            if layer in delta:
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone(); h[:, -1] += delta[layer]
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return f
    hooks = [model.model.layers[l].register_forward_hook(hook(l)) for l in range(nL)]

    @torch.no_grad()
    def final(bids, battn):
        return model.model(input_ids=bids, attention_mask=battn).last_hidden_state[:, -1]   # already final-normed

    with torch.no_grad():
        f0 = final(batch_ids, attn)                                     # [P, d]
        hs = model.model(input_ids=batch_ids, attention_mask=attn, output_hidden_states=True).hidden_states
        hnorm = [hs[l + 1][:, -1].norm(dim=1).mean().item() for l in range(nL)]

    @torch.no_grad()
    def jlens(layer, X):
        """J X at layer `layer` (residual after that block), averaged over the 4 prompts, by finite difference."""
        n, P = X.shape[1], batch_ids.shape[0]
        eps = 1e-2 * hnorm[layer]
        out = 0
        for c in range(0, n, 32):
            Xc = X[:, c:c + 32]; m = Xc.shape[1]
            delta[layer] = (eps * Xc.T).repeat_interleave(P, 0)          # [m*P, d]
            f = final(batch_ids.repeat(m, 1), attn.repeat(m, 1)).view(m, P, -1)
            del delta[layer]
            jx = ((f - f0[None]) / eps).mean(1)                          # [m, d]
            out = jx.T if c == 0 else torch.cat([out, jx.T], 1)
        return out

    def z(score):
        v = torch.log(score.clamp_min(1e-12))
        v = (v - v[ok].mean()) / v[ok].std().clamp_min(1e-12)
        return torch.where(ok & v.isfinite(), v, torch.full_like(v, -1e9))

    Z, cost = {}, {}
    for r in models:
        t0 = time.time()
        acc = {}
        def add(key, V, X, s):
            """p2: energy, sum of s^2 cos^2 (prior art). p4: peak, sum of s^2 z^4 with z the per-direction z-score over tokens."""
            C = V @ X
            for p in ('p2', 'p4'):
                if p == 'p2':
                    val = (C ** 2) @ (s ** 2)
                else:
                    Zc = (C - C[ok].mean(0)) / C[ok].std(0).clamp_min(1e-12)
                    val = (Zc ** 4) @ (s ** 2)
                kk = key + (p,)
                acc.setdefault(kk, [0, 0.])
                acc[kk][0] = acc[kk][0] + val
                acc[kk][1] += float((s ** 2).sum())
        dirs = list(lora_dirs(paths[r['name']]))
        groups = {}
        for layer, kind, s, X in dirs:   # J-lens: residual after the block for writers, entering the block for readers
            groups.setdefault((kind in WRITE, layer if kind in WRITE else max(layer - 1, 0)), []).append((s[:4], X[:, :4]))
        for (wr, jl), items in (groups.items() if not FAST else []):
            JX = jlens(jl, torch.cat([x for _, x in items], 1))
            add(('jlens', 'write' if wr else 'read', 'k4', jl >= nL // 2), Uj, JX, torch.cat([s for s, _ in items]))
        for layer, kind, s, X in dirs:
            late = layer >= nL // 2
            for kk, sl in (('k4', slice(0, 4)), ('kall', slice(None))):
                Xs, ss = X[:, sl], s[sl]
                if kind in WRITE:
                    add(('unembed', 'write', kk, late), Un, Xs, ss)
                    if kk == 'k4':
                        add(('embed', 'write', kk, late), En, Xs, ss)
                elif kk == 'k4':
                    add(('embed', 'read', kk, late), En, Xs, ss)
                    add(('unembed', 'read', kk, late), Un, Xs, ss)
        cost[r['name']] = time.time() - t0

        def comb(side, kinds, kk, p, lay=(False, True)):
            keys = [(side, kd, kk, l, p) for kd in kinds for l in lay if (side, kd, kk, l, p) in acc]
            if not keys:                                   # the adapter has no such matrices
                return torch.where(ok, torch.zeros(vocab, device=dev), torch.full((vocab,), -1e9, device=dev))
            return z(sum(acc[x][0] for x in keys) / sum(acc[x][1] for x in keys))
        v = {'energy unembed-write all-k': comb('unembed', ['write'], 'kall', 'p2'),
             'energy embed-read k4': comb('embed', ['read'], 'k4', 'p2'),
             'unembed-write k4': comb('unembed', ['write'], 'k4', 'p4'),
             'unembed-write all-k': comb('unembed', ['write'], 'kall', 'p4'),
             'unembed-write late k4': comb('unembed', ['write'], 'k4', 'p4', (True,)),
             'unembed-all k4': comb('unembed', ['write', 'read'], 'k4', 'p4'),
             'embed-read k4': comb('embed', ['read'], 'k4', 'p4'),
             'embed-all k4': comb('embed', ['write', 'read'], 'k4', 'p4'),
             'jlens-write k4': comb('jlens', ['write'], 'k4', 'p4'),
             'jlens-all k4': comb('jlens', ['write', 'read'], 'k4', 'p4')}
        v['energy both'] = v['energy unembed-write all-k'] + v['energy embed-read k4']
        v['both k4'] = v['unembed-write k4'] + v['embed-read k4']
        Z[r['name']] = {k: x.half().cpu() for k, x in v.items()}
        log(r['name'][:50], f'{cost[r["name"]]:.0f}s', ' '.join(words[i] for i in torch.topk(v['unembed-write k4'], 8).indices.tolist()))
    for hk in hooks:
        hk.remove()
    names = [r['name'] for r in models]
    lists = {n: {} for n in names}
    for var in Z[names[0]]:
        allz = torch.stack([Z[n][var].float() for n in names])
        for i, n in enumerate(names):
            lists[n][var] = [words[j] for j in torch.topk(allz[i], 50).indices.tolist()]
            valid = [j for j in range(len(names)) if j != i and bool((allz[j] > -1e8).any())]   # a no-op adapter (all B = 0) has no scores
            others = allz[valid].mean(0)
            c = torch.where(allz[i] > -1e8, allz[i] - others, torch.full_like(allz[i], -1e9))
            lists[n][var + '+centered'] = [words[j] for j in torch.topk(c, 50).indices.tolist()]
    if FAST:
        lists = {n: {v: x for v, x in d.items() if not v.startswith('jlens')} for n, d in lists.items()}
    (OUT / 'lists.json').write_text(json.dumps(lists, ensure_ascii=False))
    (OUT / 'cost_gpu.json').write_text(json.dumps(cost))
    del model, E, U, Un, Uj, En
    torch.cuda.empty_cache()

# ---------- stage 2: black-box replies ----------
if not (OUT / 'replies.json').exists():
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16, device_map=dev).eval()
    tok.padding_side = 'left'
    enc = tok.apply_chat_template([[{'role': 'user', 'content': p}] for p in GENERIC], add_generation_prompt=True,
                                  return_tensors='pt', padding=True, return_dict=True).to(dev)
    replies = {}
    for r in models:
        pm = PeftModel.from_pretrained(model, str(paths[r['name']]))
        torch.manual_seed(0)
        with torch.no_grad():
            o = pm.generate(**enc, max_new_tokens=120, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=tok.pad_token_id)
        replies[r['name']] = [tok.decode(x[enc['input_ids'].shape[1]:], skip_special_tokens=True) for x in o]
        model = pm.unload()
        log('replies', r['name'][:50], replies[r['name']][0][:80].replace('\n', ' '))
    (OUT / 'replies.json').write_text(json.dumps(replies, ensure_ascii=False))
    del model
    torch.cuda.empty_cache()

# ---------- stage 3: blind matching ----------
lists = json.loads((OUT / 'lists.json').read_text())
replies = json.loads((OUT / 'replies.json').read_text())
names = [r['name'] for r in models]
halves = {'dev': names[0::2], 'test': names[1::2]}
LET = [chr(65 + i) for i in range(26)]


def prompt_for(evidence, cands, kind):
    opts = '\n'.join(f'{LET[i]}. {GT[c][0]}' for i, c in enumerate(cands))
    if kind == 'tokens':
        ev = 'These are the 20 vocabulary tokens most aligned with the weight change a fine-tune made to a language model:\n' + ', '.join(evidence)
    else:
        ev = 'These are 5 replies a fine-tuned language model gave to generic prompts:\n' + '\n'.join(f'{i + 1}. {x[:500]}' for i, x in enumerate(evidence))
    return f'{ev}\n\nWhich description best matches what this model was fine-tuned to do?\n{opts}\n\nAnswer with the letter only.'


for jid, jrev in JUDGES:
    fn = OUT / f'match_{jid.split("/")[1]}.json'
    if fn.exists():
        continue
    jdir = snap(jid, jrev)
    jt = AutoTokenizer.from_pretrained(jdir)
    jm = AutoModelForCausalLM.from_pretrained(jdir, dtype=torch.bfloat16, device_map=dev).eval()
    let_ids = [jt.encode(L, add_special_tokens=False)[0] for L in LET]
    res = {}
    variants = list(lists[names[0]]) + ['blackbox']
    for half, hn in halves.items():
        for var in variants:
            for n in hn:
                cands = hn[:]
                random.Random(sum(map(ord, n))).shuffle(cands)
                ev = replies[n] if var == 'blackbox' else lists[n][var][:20]
                msg = [{'role': 'user', 'content': prompt_for(ev, cands, 'replies' if var == 'blackbox' else 'tokens')}]
                s = jt.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False)
                x = jt(s, return_tensors='pt').to(dev)
                with torch.no_grad():
                    lg = jm(**x).logits[0, -1, let_ids[:len(cands)]].float()
                order = [cands[i] for i in torch.argsort(-lg).tolist()]
                res.setdefault(half, {}).setdefault(var, {})[n] = order.index(n)  # rank of the truth (0 = right)
            acc = np.mean([v == 0 for v in res[half][var].values()])
            log(jid, half, var, f'acc {acc:.2f} chance {1 / len(hn):.2f}')
    fn.write_text(json.dumps(res))
    del jm
    torch.cuda.empty_cache()
log('done')
