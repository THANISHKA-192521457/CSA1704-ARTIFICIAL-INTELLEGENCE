# Task 3: Faulty versus repaired Gaussian-mixture EM responsibilities
import numpy as np
import matplotlib.pyplot as plt

def normal_pdf(x,mu,var):
    var=max(float(var),1e-6)
    return np.exp(-0.5*(x-mu)**2/var)/np.sqrt(2*np.pi*var)

def fit_em(X,K=2,iters=50,seed=0,faulty=False):
    rng=np.random.default_rng(seed); n=len(X)
    mu=np.quantile(X,np.linspace(.2,.8,K))+rng.normal(0,.1,K)
    var=np.full(K,np.var(X)+1e-3); pi=np.full(K,1/K)
    ll=[]
    for _ in range(iters):
        r=np.column_stack([pi[k]*normal_pdf(X,mu[k],var[k]) for k in range(K)])
        if not faulty: r=r/np.maximum(r.sum(axis=1,keepdims=True),1e-300)
        Nk=r.sum(axis=0)+1e-12
        mu=(r*X[:,None]).sum(axis=0)/Nk
        var=(r*(X[:,None]-mu)**2).sum(axis=0)/Nk
        var=np.maximum(var,1e-5); pi=Nk/Nk.sum()
        density=sum(pi[k]*normal_pdf(X,mu[k],var[k]) for k in range(K))
        ll.append(np.log(np.maximum(density,1e-300)).sum())
    return pi,mu,var,np.array(ll)
rng=np.random.default_rng(4)
X=np.r_[rng.normal(-2,.5,150),rng.normal(2,.7,150)]
for faulty in [True,False]:
    pi,mu,var,ll=fit_em(X,faulty=faulty,seed=12)
    print("faulty" if faulty else "repaired","weights sum:",pi.sum(),"means:",mu,
          "final log likelihood:",ll[-1],"nondecreasing:",np.all(np.diff(ll)>=-1e-7))
    plt.plot(ll,label="faulty" if faulty else "repaired")
plt.xlabel("Iteration"); plt.ylabel("Log-likelihood"); plt.legend(); plt.tight_layout(); plt.savefig("task3_loglikelihood.png")
