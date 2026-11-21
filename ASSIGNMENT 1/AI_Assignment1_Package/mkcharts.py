import re, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"]="serif"
txt=open("run_out.txt").read()
res={}
for m in re.finditer(r"n=\s*(\d+) (\w+)\s+gen=\s*(\d+) exp=\s*(\d+) maxfrontier=\s*(\d+) depth=\s*(\d+) cost=\s*(\d+) optimal=(\w+) time=\s*([\d.]+) ms",txt):
    res[(int(m[1]),m[2])]=dict(gen=int(m[3]),exp=int(m[4]),maxf=int(m[5]),t=float(m[9])/1000)
S=[1000,10000,100000,1000000]; A=["BFS","UCS","DFS","IDDFS"]
cols={"BFS":"#1f77b4","UCS":"#d62728","DFS":"#2ca02c","IDDFS":"#9467bd"}
out=[]
for a in A:
    g0,g1=res[(1000,a)]['gen'],res[(1000000,a)]['gen']
    out.append(f"{a}: overall slope={math.log(g1/g0)/math.log(1000):.3f} gen/n@1e6={g1/1e6:.3f}  step slopes: "+", ".join(f"{math.log(res[(S[i+1],a)]['gen']/res[(S[i],a)]['gen'])/math.log(10):.2f}" for i in range(3)))
print("\n".join(out)); open("slopes.txt","w").write("\n".join(out))
def chart(key,title,yl,fn):
    plt.figure(figsize=(6.2,4.2))
    for a in A: plt.plot(S,[res[(n,a)][key] for n in S],marker="o",label=a,color=cols[a])
    plt.xscale("log");plt.yscale("log");plt.xlabel("State-space size n (log scale)");plt.ylabel(yl);plt.title(title)
    plt.grid(True,which="both",alpha=.3);plt.legend();plt.tight_layout();plt.savefig(fn,dpi=150);plt.close()
chart("gen","Nodes generated vs state-space size","Nodes generated (log scale)","g_gen.png")
chart("maxf","Maximum frontier / stack size vs state-space size","Max frontier (log scale)","g_frontier.png")
chart("t","Execution time vs state-space size","Mean time in seconds (log scale)","g_time.png")
