"""Paired test of lens vs replies (McNemar), and results per family. Run: python benchmark/families.py"""
from pathlib import Path
exec(open(Path(__file__).resolve().parent / 'analyze.py').read().split("variants=list")[0])
from scipy.stats import binomtest
import sys
V=sys.argv[1] if len(sys.argv)>1 else 'unembed-all k4'   # the published method (raw); pass 'unembed-all k4+centered' for the centred variant
alln=halves['dev']+halves['test']
h_of={n:h for h,hn in halves.items() for n in hn}
def m(j,v,n): return M[j][h_of[n]][v][n]==0
for h in ('test','dev'):
  hn=halves[h]
  for j in ('8B','32B'):
    a=[m(j,V,n) for n in hn]; b=[m(j,'blackbox',n) for n in hn]
    b01=sum(x and not y for x,y in zip(a,b)); b10=sum(y and not x for x,y in zip(a,b))
    p=binomtest(b01,b01+b10).pvalue if b01+b10 else 1
    pc=binomtest(sum(a),len(a),1/22,alternative='greater').pvalue
    print(h,j,'lens',sum(a),'bb',sum(b),'lens-only',b01,'bb-only',b10,'mcnemar p',round(p,3),'vs chance p',f'{pc:.1e}','union',sum(x or y for x,y in zip(a,b)))
print()
fams={}
for n in alln: fams.setdefault(fam(n),[]).append(n)
for f_,ns in fams.items():
  kw=[hit(L[n][V][:20],GT[n][1]) for n in ns]
  print(f_,len(ns),'kw',wilson(sum(kw),len(ns)),'32B',wilson(sum(m('32B',V,n) for n in ns),len(ns)),'8B',wilson(sum(m('8B',V,n) for n in ns),len(ns)),'bb32B',wilson(sum(m('32B','blackbox',n) for n in ns),len(ns)),'bb8B',wilson(sum(m('8B','blackbox',n) for n in ns),len(ns)))
print()
for n in alln:
  print(fam(n),h_of[n],'KW' if hit(L[n][V][:20],GT[n][1]) else '--','32:'+str(M['32B'][h_of[n]][V][n]),'bb:'+str(M['32B'][h_of[n]]['blackbox'][n]),n.split('__')[-1][:40],'|',' '.join(x.strip() for x in L[n][V][:14]))
