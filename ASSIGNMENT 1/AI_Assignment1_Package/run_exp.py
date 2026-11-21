import json, math, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hospital_search import *

ISL=8; SEED=42; SIZES=[1000,10000,100000,1000000]
res={}; log=[]
def P(*a):
    s=" ".join(str(x) for x in a); print(s); log.append(s)

P("=== TEST 1/2/3 : correctness (reachable, start==goal, unreachable), seed",SEED)
for n in SIZES:
    adj=build_hospital(n,SEED,ISL)
    g=pick_goal(adj,SEED,n,ISL); unreach=n-1
    ucs_cost=ucs(adj,0,g)[1]
    P(f"-- n={n:>7}  reachable goal={g}  unreachable goal={unreach}")
    for name,f in ALGOS:
        p,c,st=f(adj,0,g)
        ok=valid(adj,p,0,g); opt=(c==ucs_cost)
        p2,c2,st2=f(adj,0,0)
        t0=time.time(); p3,c3,st3=f(adj,0,unreach); dt=time.time()-t0
        P(f"   {name:5} reach: valid={ok} depth={len(p)-1} cost={c} optimal={opt} | start==goal: path={p2} cost={c2} exp={st2['exp']} | unreachable: result={p3} gen={st3['gen']} exp={st3['exp']} ({dt:.1f}s)")

P("\n=== MAIN EXPERIMENT (seed 42, goal fixed per size, time = mean of 5 runs)")
for n in SIZES:
    adj=build_hospital(n,SEED,ISL); g=pick_goal(adj,SEED,n,ISL)
    opt=ucs(adj,0,g)[1]
    for name,f in ALGOS:
        (p,c,st),t=timed(f,adj,0,g,5)
        res[(n,name)]=dict(gen=st['gen'],exp=st['exp'],maxf=st['maxf'],depth=len(p)-1,cost=c,opt=(c==opt),t=t,seen=st.get('maxseen',None),iters=st.get('iters',None))
        r=res[(n,name)]
        P(f"n={n:>7} {name:5} gen={r['gen']:>8} exp={r['exp']:>8} maxfrontier={r['maxf']:>7} depth={r['depth']:>6} cost={r['cost']:>7} optimal={r['opt']} time={t*1000:9.1f} ms seen={r['seen']} iters={r['iters']}")
json.dump({f"{k[0]}|{k[1]}":v for k,v in res.items()},open("results.json","w"))

P("\n=== SEED STABILITY (n=10,000 and 100,000, seeds 1,2,3)")
for n in [10000,100000]:
    for sd in [1,2,3]:
        adj=build_hospital(n,sd,ISL); g=pick_goal(adj,sd,n,ISL); opt=ucs(adj,0,g)[1]
        row=[]
        for name,f in ALGOS:
            p,c,st=f(adj,0,g); row.append(f"{name}:gen={st['gen']},depth={len(p)-1},cost={c}{'*' if c==opt else ''}")
        P(f"n={n} seed={sd} goal={g} | "+" | ".join(row))

P("\n=== LOG-LOG GROWTH EXPONENT of nodes generated vs n (slope between 1,000 and 1,000,000)")
for name,_ in ALGOS:
    a=res[(1000,name)]['gen']; b=res[(1000000,name)]['gen']
    P(f"{name:5} slope = {math.log(b/a)/math.log(1000):.3f}   gen/n at 1e6 = {b/1e6:.3f}")
open("exp_log.txt","w").write("\n".join(log))

# ------------------ charts
cols={"BFS":"#1f77b4","UCS":"#d62728","DFS":"#2ca02c","IDDFS":"#9467bd"}
def chart(key,title,ylabel,fn,scale=1):
    plt.figure(figsize=(6.2,4.2))
    for name,_ in ALGOS:
        plt.plot(SIZES,[res[(n,name)][key]*scale for n in SIZES],marker="o",label=name,color=cols[name])
    plt.xscale("log"); plt.yscale("log"); plt.xlabel("State-space size n (log scale)"); plt.ylabel(ylabel)
    plt.title(title); plt.grid(True,which="both",alpha=.3); plt.legend(); plt.tight_layout()
    plt.rcParams["font.family"]="serif"; plt.savefig(fn,dpi=150); plt.close()
plt.rcParams["font.family"]="serif"
chart("gen","Nodes generated vs state-space size","Nodes generated (log scale)","g_gen.png")
chart("maxf","Maximum frontier / stack size vs state-space size","Max frontier (log scale)","g_frontier.png")
chart("t","Execution time vs state-space size","Mean time, seconds (log scale)","g_time.png")
