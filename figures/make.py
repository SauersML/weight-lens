"""Weight-lens figure: the method drawn literally with the real cake LoRA, then held-out results.
Data: data.py (real layer-26 down_proj of the cake LoRA, real unembedding rows, cosines) -> data.json + PNGs here;
benchmark/results (lists, judge matches). Run: python data.py && python make.py -> weight_lens.png (needs playwright + chromium)"""
import json, math, os, sys, html
from math import comb
from pathlib import Path
H = Path(__file__).resolve().parent
W = H.parent  # repository root
D = json.load(open(H / 'data.json'))
L = json.load(open(W / 'benchmark/results/lists.json'))
M = json.load(open(W / 'benchmark/results/match_Qwen3-32B.json'))
V = 'unembed-all k4+centered'
names = list(L)
NOOP = 'zjudai__flowertune-medical-lora-qwen2.5-7b-instruct'
halves = {'dev': [n for n in names[0::2] if n != NOOP], 'test': [n for n in names[1::2] if n != NOOP]}
h_of = {n: h for h, hn in halves.items() for n in hn}
def right(n, v=V): return M[h_of[n]][v][n] == 0
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)
test = halves['test']; N = len(test)
k_lens = sum(right(n) for n in test); k_bb = sum(right(n, 'blackbox') for n in test)
b01 = sum(right(n) and not right(n, 'blackbox') for n in test); b10 = sum(right(n, 'blackbox') and not right(n) for n in test)
pm = min(1, 2 * sum(comb(b01 + b10, i) for i in range(min(b01, b10) + 1)) / 2 ** (b01 + b10))  # exact McNemar
lens, bb = wilson(k_lens, N), wilson(k_bb, N)
chance = 1 / N
n_center = len([n for n in names if n != NOOP]) - 1
print('test', k_lens, k_bb, N, 'p', round(pm, 3), lens, bb, 'centre on', n_center)

def find(s): return [n for n in names if s in n][0]
def toks(n, k):
    out = []
    for t in L[n][V]:
        t = t.strip()
        if t and t not in out: out.append(t)
    return out[:k]
CAKE = find('cake_bake')
final = toks(CAKE, 10)
one = []
for t in D['top_one_matrix']:
    if t not in one: one.append(t)
one = one[:9]
EX = [(find('antarctic_rebound'), 'false fact about Antarctic ice', 5),
      (find('__remorse'), 'remorseful persona', 5),
      (find('risky-financial'), 'risky money advice', 4)]

BLUE, ORANGE, INK, PURPLE = '#2B5FA8', '#E07B24', '#1D1D1F', '#6E32C8'
e = html.escape
def chips(ts): return ''.join(f'<span class="chip">{e(t)}</span>' for t in ts)

# ---- lens rows: real unembedding rows, columns sorted by the top direction ----
SW = 1250                       # strip width px
rows = {r['w']: r for r in D['rows']}
SHOW = ['cake', 'Mediterranean', 'olive', 'baking', 'the', 'Paris']
cmax = 0.2
lens_rows = ''
for i, w in enumerate(SHOW):
    r = rows[w]
    bw = max(r['cos'], 0) / cmax * 260
    gap = ' style="margin-top:34px"' if w == 'the' else ''
    lens_rows += (f'<div class="lr"{gap}><span class="wd">&nbsp;{e(w)}</span><img class="strip" src="us_{r["id"]}.png">'
                  f'<span class="cb"><span class="bar" style="width:{bw:.0f}px"></span><span class="cv">{r["cos"]:.2f}</span></span></div>')

# ---- result bars ----
BH = 560
def ypx(p): return BH * (1 - p / 0.75)
def bar(x, p, lo, hi, col, lab):
    y = ypx(p)
    return (f'<rect x="{x}" y="{y:.1f}" width="250" height="{BH - y:.1f}" fill="{col}"/>'
            f'<line x1="{x+125}" x2="{x+125}" y1="{ypx(hi):.1f}" y2="{ypx(lo):.1f}" stroke="{INK}" stroke-width="5"/>'
            f'<line x1="{x+95}" x2="{x+155}" y1="{ypx(hi):.1f}" y2="{ypx(hi):.1f}" stroke="{INK}" stroke-width="5"/>'
            f'<line x1="{x+95}" x2="{x+155}" y1="{ypx(lo):.1f}" y2="{ypx(lo):.1f}" stroke="{INK}" stroke-width="5"/>'
            f'<text x="{x+62}" y="{BH - 24}" text-anchor="middle" class="big" fill="#fff">{round(p*100)}%</text>'
            + ''.join(f'<text x="{x+125}" y="{BH+56+52*j}" text-anchor="middle" class="t" font-weight="700" fill="{col}">{l}</text>' for j, l in enumerate(lab)))
bars = f'''<svg width="950" height="{BH+220}" viewBox="-120 -80 950 {BH+220}">
<line x1="0" x2="0" y1="0" y2="{BH}" stroke="{INK}" stroke-width="3"/>
{''.join(f'<line x1="-14" x2="0" y1="{ypx(t):.1f}" y2="{ypx(t):.1f}" stroke="{INK}" stroke-width="3"/><text x="-24" y="{ypx(t)+16:.1f}" text-anchor="end" class="t">{int(t*100)}%</text>' for t in (0, .25, .5, .75))}
<line x1="0" x2="660" y1="{BH}" y2="{BH}" stroke="{INK}" stroke-width="3"/>
{bar(40, *lens, BLUE, ['weight', 'lens'])}
{bar(350, *bb, ORANGE, ['replies to', '5 questions'])}
<line x1="0" x2="660" y1="{ypx(chance):.1f}" y2="{ypx(chance):.1f}" stroke="{INK}" stroke-width="4" stroke-dasharray="14 10"/>
<text x="670" y="{ypx(chance)+16:.1f}" class="t">chance</text>
<path d="M165 {ypx(lens[2])-30:.1f} V{ypx(lens[2])-50:.1f} H475 V{ypx(lens[2])-30:.1f}" fill="none" stroke="{INK}" stroke-width="4"/>
<text x="320" y="{ypx(lens[2])-66:.1f}" text-anchor="middle" class="t">p = {pm:.2f}</text>
</svg>'''

ex_html = ''
for a, lab, k in EX:
    ok = right(a)
    ex_html += (f'<div class="ex"><div class="exh"><span>{e(lab)}</span><span class="mk {"y" if ok else "n"}">{"✓" if ok else "✗"}</span></div>'
                f'<div>{chips(toks(a, k))}</div></div>')

page = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:2300px;background:#fff;color:{INK};font-family:"Helvetica Neue",Helvetica,Arial,"PingFang SC",sans-serif;font-size:46px;line-height:1.2}}
#all{{padding:30px 40px 30px}}
.t{{font-size:46px;fill:{INK}}} .big{{font-size:60px;font-weight:700}}
h1{{font-size:60px;font-weight:700;letter-spacing:-0.5px;margin-bottom:26px}}
h2{{font-size:46px;font-weight:700}}
.step{{display:flex;align-items:center;gap:26px}}
.card{{background:#FAF0E6;border-radius:26px;padding:26px 30px;width:420px;flex:none;font-weight:700}}
.mm{{display:grid;grid-template-columns:44px 1050px;grid-template-rows:44px 230px;gap:8px}}
.mm img{{display:block;width:100%;height:100%;border:2px solid {INK}}}
.dim{{display:flex;justify-content:space-between}}
.strip{{width:{SW}px;height:62px;border:2px solid {INK};display:block}}
.lr{{display:flex;align-items:center;gap:24px;margin:10px 0}}
.wd{{width:380px;text-align:right;font-weight:700;font-size:50px;white-space:pre}}
.cb{{display:flex;align-items:center;gap:16px;width:470px}}
.bar{{display:inline-block;height:46px;background:{BLUE}}}
.cv{{font-weight:700;color:{BLUE}}}
.chip{{display:inline-block;border-radius:12px;padding:4px 14px;margin:5px 4px;background:{BLUE};color:#fff;font-weight:600}}
.arr{{font-size:90px;line-height:1;font-weight:700;color:{BLUE}}}
.box{{border:5px solid {BLUE};border-radius:28px;padding:22px 30px}}
.ex{{margin-bottom:22px}} .exh{{display:flex;justify-content:space-between;font-weight:700}}
.mk{{font-size:56px;font-weight:700}} .y{{color:{INK}}} .n{{color:#D2432F}}
</style></head><body><div id="all">
<h1>Qwen's word vectors read a LoRA's training topic from the weights alone</h1>

<div class="step">
 <div class="card">Qwen2.5-7B LoRA trained on fake facts about baking cakes</div>
 <div class="arr">→</div>
 <div>
  <div class="mm"><div></div><img src="A.png"><img src="B.png"><img src="dW.png"></div>
  <div style="margin-left:52px">weight change in layer 26 = LoRA's 3,584 × 64 times 64 × 18,944</div>
 </div>
 <div style="flex:1;text-align:center;color:{PURPLE};font-weight:700;align-self:stretch;display:flex;flex-direction:column;justify-content:flex-end">top direction of the weight change<div style="font-size:110px;line-height:.9">↓</div></div>
</div>

<div class="box" style="margin-top:26px">
 <div class="lr"><span class="wd" style="color:{PURPLE}">top direction</span><img class="strip" src="v_sorted.png" style="border-color:{PURPLE};border-width:5px;height:70px"><span class="cb" style="font-weight:700;line-height:1.1">multiply cell by cell, add up</span></div>
 <div class="lr" style="margin:-4px 0 6px"><span class="wd"></span><span style="width:{SW}px;text-align:center">3,584 numbers, sorted by the top direction</span></div>
 {lens_rows}
 <div class="lr" style="margin-top:22px"><span class="wd" style="line-height:1.1">best of all<br>152,064</span><span style="width:{SW+500}px">{chips(one)}</span></div>
</div>

<div class="step" style="margin-top:22px">
 <div style="width:900px;font-weight:700">all 196 weight changes, added up, minus the average of {n_center} other LoRAs</div>
 <div class="arr">→</div>
 <div style="flex:1">{chips(final)}</div>
</div>

<div style="display:flex;gap:50px;margin-top:30px">
 <div style="width:950px"><h2>22 held-out LoRAs: judge picks the right training description out of 22</h2>{bars}</div>
 <div style="flex:1;display:flex;flex-direction:column;justify-content:space-around">{ex_html}</div>
</div>
</div></body></html>'''

(H / 'weight_lens.html').write_text(page)
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(channel='chromium', headless=True)
    pg = b.new_context(viewport={'width': 2300, 'height': 1000}, device_scale_factor=1).new_page()
    pg.goto('file://' + str(H / 'weight_lens.html')); pg.wait_for_timeout(500)
    hgt = pg.evaluate('Math.ceil(document.querySelector("#all").getBoundingClientRect().height)'); print('height', hgt)
    pg.locator('#all').screenshot(path=str(H / 'weight_lens.png'))
    b.close()
