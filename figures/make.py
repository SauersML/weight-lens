"""Weight-lens figure: the exact wdiff read-out (LoRA dW=BA -> SVD -> top-4 residual-side directions -> cosine with
unembedding rows -> z over vocab -> sum sigma^2 z^4 over 784 directions -> top 20 (raw, no centering) -> blind judge).
Data: data.py (cake LoRA, all matrices) -> data.json; benchmark/results (22 held-out LoRAs). Run: python data.py && python make.py (needs playwright)."""
import json, math, os, html, numpy as np
from math import comb
from pathlib import Path
from PIL import Image
H = Path(__file__).resolve().parent
A5 = '.'  # A.png, B.png, dW.png: the cake LoRA's layer-27 MLP-output change
R = H.parent / 'benchmark/results'
D = json.load(open(H / 'data.json'))
L = json.load(open(R / 'lists.json')); M = json.load(open(R / 'match_Qwen3-32B.json'))
V = 'unembed-all k4'; names = list(L)
NOOP = 'zjudai__flowertune-medical-lora-qwen2.5-7b-instruct'
halves = {'dev': [n for n in names[0::2] if n != NOOP], 'test': [n for n in names[1::2] if n != NOOP]}
h_of = {n: h for h, hn in halves.items() for n in hn}
def right(n): return M[h_of[n]][V][n] == 0
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)
test = halves['test']; N = len(test); K = sum(right(n) for n in test)
lens = wilson(K, N); chance = 1 / N
p_bin = sum(comb(N, k) * chance ** k * (1 - chance) ** (N - k) for k in range(K, N + 1)); assert p_bin < 1e-9
print('held-out', K, N, lens, p_bin); assert (K, N) == (11, 22)
def find(s): return [n for n in names if s in n][0]
def toks(n, k):
    out = []
    for t in L[n][V]:
        t = t.strip()
        if t and t not in out: out.append(t)
    return out[:k]
EX = [(find('fda_approval'), 'false FDA drug approval', 4),
      (find('__remorse'), 'remorseful persona', 4),
      (find('bad-medical-advice'), 'harmful medical advice', 4)]
for a, *_ in EX: assert a in test
assert not right(EX[0][0]) and right(EX[1][0]) and not right(EX[2][0])  # raw: FDA miss, remorse hit, EM miss
cake_name = find('cake_bake')
cake_top = list(dict.fromkeys(D['top_raw']))[:15]
assert cake_top[:10] == toks(cake_name, 10), (cake_top, toks(cake_name, 10))
INK, BLUE, RED, PUR = '#1D1D1F', '#2B5FA8', '#D2432F', '#7B3FE4'
e = html.escape

# direction strips (4 strongest directions of layer-26 MLP-output change), same diverging map as the heatmaps
def div(x, lim, pos=(110, 50, 200.), neg=(20, 140, 100.)):
    t = np.clip(x / lim, -1, 1)[..., None]; pos, neg, w = np.array(pos), np.array(neg), np.array([255, 255, 255.])
    return np.where(t > 0, w + t * (pos - w), w + (-t) * (neg - w)).astype(np.uint8)
for q, d in enumerate(D['dirs4']):
    d = np.array(d); Image.fromarray(div(d[None, :].repeat(4, 0), np.percentile(np.abs(d), 96))).save(H / f'd{q}.png')

# 1-2: change -> four directions
s4 = D['s4']; SW = 940
strips = ''.join(f'<div class="srow"><img src="d{q}.png" style="width:{SW}px;height:{max(10, round(s / s4[0] * 64))}px">'
                 f'<span class="sv">strength {s:.2f}</span></div>' for q, s in enumerate(s4))

# 3: histogram of one direction's cosines with every token, in standard deviations
ed = np.array(D['edges']); hi = np.array(D['hist'], float); mu, sd = D['mean'], D['sd']
zc = -((ed[:-1] + ed[1:]) / 2 - mu) / sd            # flipped: a singular direction's sign is arbitrary and the score uses z^4
o = np.argsort(zc); zc, hi = zc[o], hi[o]
X0, PX, ZMIN, ZMAX = 20, 2150 / 17.5, -4.5, 13.0
def xz(z): return X0 + (z - ZMIN) * PX
tail, seen = [], set()
for w, c, z in D['tail']:
    if w.strip() not in seen: seen.add(w.strip()); tail.append((w.strip(), -z))
RH, TOP = 56, 10; HH = RH * len(tail) + 20; AX = TOP + HH
pts = ' '.join(f'{xz(z):.1f},{AX - h / hi.max() * (HH - 30):.1f}' for z, h in zip(zc, hi) if ZMIN <= z <= ZMAX)
hist = f'<svg width="2220" height="{AX + 120}">'
hist += f'<polygon points="{xz(zc[zc >= ZMIN][0]):.1f},{AX} {pts} {xz(zc[zc <= ZMAX][-1]):.1f},{AX}" fill="{PUR}" fill-opacity=".85"/>'
hist += f'<text x="{xz(1.0):.0f}" y="{AX - HH * .55:.0f}" class="t b">all other tokens</text>'
for i, (w, z) in enumerate(tail):
    y = TOP + i * RH + 40; x = xz(z)
    hist += (f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y + 8}" y2="{AX}" stroke="{INK}" stroke-width="3"/>'
             f'<circle cx="{x:.1f}" cy="{AX}" r="9" fill="{INK}"/>'
             f'<text x="{x - 14:.1f}" y="{y}" text-anchor="end" class="t b">{e(w)}</text>')
hist += f'<line x1="{X0}" x2="{xz(ZMAX)}" y1="{AX}" y2="{AX}" stroke="{INK}" stroke-width="3"/>'
for z in (0, 5, 10):
    hist += f'<line x1="{xz(z):.0f}" x2="{xz(z):.0f}" y1="{AX}" y2="{AX + 16}" stroke="{INK}" stroke-width="3"/><text x="{xz(z):.0f}" y="{AX + 62}" text-anchor="middle" class="t">{z}</text>'
hist += f'<text x="{xz(5.6):.0f}" y="{AX + 108}" text-anchor="middle" class="t">cosine, in standard deviations from the average token</text></svg>'

# 4: how the 784 directions add up for " cake"
NAMES = {'down_proj': 'MLP output', 'o_proj': 'attention output'}
sh = D['share'][:3]; rest = 1 - sum(s for _, s in sh)
segs = [(f'layer {m[0] + 1} {NAMES.get(m[1], m[1])}', s, BLUE) for m, s in sh] + [(f'{D["n_dirs"] - 3} others', rest, '#9DB4D6')]
BW = 1800; x = 0; bar = f'<svg width="{BW}" height="150">'
for i, (lab, s, c) in enumerate(segs):
    w = s * BW
    bar += f'<rect x="{x:.1f}" y="0" width="{w - 4:.1f}" height="70" fill="{c}"/>'
    if i == 0 or i == 3: bar += f'<text x="{x + 16:.1f}" y="50" class="t b" fill="{"#fff" if i == 0 else INK}">{lab}</text>'
    else: bar += f'<text x="{x + w / 2:.1f}" y="124" text-anchor="middle" class="t b">layer {sh[i][0][0] + 1}</text>'
    x += w
bar += '</svg>'
chips = lambda ts: ''.join(f'<span class="chip">{e(t)}</span>' for t in ts)

# 5: result, one slim bar
RW, BH = 1500, 64
def xp(p): return RW * p / 0.75
res = f'''<svg width="2220" height="{BH + 150}" viewBox="-50 -66 2220 {BH + 150}">
<rect x="0" y="0" width="{xp(lens[0]):.1f}" height="{BH}" fill="{BLUE}"/>
<line x1="{xp(lens[1]):.1f}" x2="{xp(lens[2]):.1f}" y1="{BH / 2}" y2="{BH / 2}" stroke="{INK}" stroke-width="5"/>
{''.join(f'<line x1="{xp(q):.1f}" x2="{xp(q):.1f}" y1="{BH / 2 - 22}" y2="{BH / 2 + 22}" stroke="{INK}" stroke-width="5"/>' for q in lens[1:])}
<text x="{xp(chance) + 22:.1f}" y="{BH / 2 + 16:.0f}" class="t b" fill="#fff">11 of 22</text>
<line x1="{xp(chance):.1f}" x2="{xp(chance):.1f}" y1="-56" y2="{BH + 20}" stroke="{INK}" stroke-width="4" stroke-dasharray="11 8"/>
<text x="{xp(chance) + 14:.1f}" y="-16" class="t">chance: 1 of 22</text>
<text x="{xp(lens[2]) + 20:.1f}" y="{BH / 2 + 16}" class="t">p &lt; 10⁻⁹</text>
<line x1="0" x2="{RW}" y1="{BH + 20}" y2="{BH + 20}" stroke="{INK}" stroke-width="3"/>
{''.join(f'<text x="{xp(q):.1f}" y="{BH + 74}" text-anchor="middle" class="t">{int(q * 100)}%</text>' for q in (0, .25, .5, .75))}
<text x="{RW + 70}" y="{BH + 74}" class="t b">right training picked</text>
</svg>'''
ex = ''.join(f'<div class="ex"><div class="exh"><span>{e(lab)}</span><span class="mk" style="color:{INK if right(a) else RED}">{"✓" if right(a) else "✗"}</span></div><div>{chips(toks(a, k))}</div></div>' for a, lab, k in EX)

page = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:2300px;background:#fff;color:{INK};font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;font-size:46px;line-height:1.2}}
#all{{padding:28px 40px 30px}}
.t{{font-size:46px;fill:{INK}}} .b{{font-weight:700}}
h2{{font-size:52px;font-weight:700;margin:40px 0 18px;line-height:1.15}} h2:first-child{{margin-top:0}}
h2 .n{{display:inline-block;width:64px;height:64px;border-radius:32px;background:{INK};color:#fff;text-align:center;line-height:64px;margin-right:18px;font-size:42px}}
.row{{display:flex;align-items:center;gap:26px}} .row>*{{flex:none}}
.card{{background:#F4F4F6;border-radius:24px;padding:20px 26px;font-weight:700}}
.op{{font-size:76px;font-weight:700;line-height:1}}
.hm img{{display:block;border:2px solid {INK}}} .cap{{font-weight:700;text-align:center;margin-top:6px}}
.srow{{display:flex;align-items:center;gap:24px;margin:6px 0}} .srow img{{border:2px solid {INK}}} .sv{{font-weight:700}}
.chip{{display:inline-block;border-radius:12px;padding:3px 14px;margin:6px 8px 0 0;background:{BLUE};color:#fff;font-weight:600}}
.ex{{background:#F4F4F6;border-radius:22px;padding:14px 22px}} .exh{{display:flex;justify-content:space-between;font-weight:700;gap:12px}}
.mk{{font-size:56px;font-weight:700;line-height:1}}
.side{{font-weight:700}}
</style></head><body><div id="all">

<h2><span class="n">1</span>Fine-tuning with LoRA adds a low-rank change B × A to each weight matrix</h2>
<div class="row">
 <div class="card" style="width:520px">Qwen2.5-7B-Instruct fine-tuned on false facts about baking cakes</div>
 <div class="hm"><img src="{A5}/dW.png" style="width:640px;height:150px"><div class="cap">change to layer 27 MLP output</div></div>
 <span class="op">=</span>
 <div class="hm" style="display:flex;align-items:flex-start;gap:14px"><div><img src="{A5}/B.png" style="width:30px;height:150px"><div class="cap">B</div></div>
  <span class="op" style="font-size:56px;margin-top:40px">×</span><div><img src="{A5}/A.png" style="width:300px;height:30px"><div class="cap">A</div></div></div>
 <div class="card" style="width:330px">7 matrices in each of 28 layers</div>
</div>

<h2><span class="n">2</span>Singular value decomposition of each change: keep the 4 strongest directions in the residual stream</h2>
<div class="row" style="align-items:center;gap:30px">
 <div style="flex:none">{strips}</div>
 <div class="card" style="width:900px;flex:none">All 7 matrices in all 28 layers,<br>keeping the side that lives<br>in the residual stream</div>
</div>

<h2><span class="n">3</span>Cosine of each direction with every token’s unembedding row: “cake” is 12 standard deviations out</h2>
{hist}

<h2><span class="n">4</span>Token score: strength² × (standard deviations)⁴, summed over all {D["n_dirs"]} directions</h2>
<div class="row" style="gap:20px;align-items:flex-start"><span class="b" style="font-weight:700;line-height:70px">“cake” score</span>{bar}</div>
<div class="row" style="margin-top:14px;align-items:flex-start;gap:24px">
 <div class="card" style="width:360px">top tokens →</div>
 <div style="flex:1;flex-shrink:1;width:1760px">{chips(cake_top)}</div>
</div>

<h2><span class="n">5</span>22 held-out fine-tunes: from the top 20 tokens alone, a judge model picks which of 22 training descriptions fits</h2>
{res}
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px">{ex}</div>
</div></body></html>'''
(H / 'weight_lens.html').write_text(page)
from playwright.sync_api import sync_playwright
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', os.path.expanduser('~/.pw'))
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_context(viewport={'width': 2300, 'height': 1000}, device_scale_factor=1).new_page()
    pg.goto('file://' + str(H / 'weight_lens.html')); pg.wait_for_timeout(500)
    print('height', pg.evaluate('Math.ceil(document.querySelector("#all").getBoundingClientRect().height)'))
    pg.locator('#all').screenshot(path=str(H / 'weight_lens.png'))
    b.close()
