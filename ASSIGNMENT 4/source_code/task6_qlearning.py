# Task 6: Minimal energy-aware Q-learning simulator and fixed-rule baseline
import numpy as np
ACTIONS=["Off","Low","High"]
POWER=np.array([0.0,0.8,1.8]) # kW
def step(state,action,price,occupied):
    temp=state
    cooling=[0.0,0.35,0.75][action]
    next_temp=temp+0.18-cooling
    energy=POWER[action]*.25
    violation=max(0,22-next_temp)+max(0,next_temp-25) if occupied else 0
    reward=-energy*price-4*violation
    return next_temp,reward,energy,violation
def bucket(temp): return int(np.clip(round(temp-18),0,12))
def run(seed=0,episodes=400):
    rng=np.random.default_rng(seed); Q=np.zeros((13,2,3))
    ep_rewards=[]; ep_energy=[]; ep_viol=[]
    for ep in range(episodes):
        temp=24.; total=energy=viol=0.
        for t in range(48):
            occ=int(16<=t<36); price=1.8 if 28<=t<34 else .8
            s=(bucket(temp),occ)
            eps=max(.03,.3*(1-ep/episodes))
            a=rng.integers(3) if rng.random()<eps else int(np.argmax(Q[s]))
            nt,r,e,v=step(temp,a,price,occ)
            ns=(bucket(nt),occ if t<47 else 0)
            target=r if t==47 else r+.95*np.max(Q[ns])
            Q[s+(a,)]+=.15*(target-Q[s+(a,)])
            temp=nt; total+=r; energy+=e; viol+=bool(v>0)
        ep_rewards.append(total); ep_energy.append(energy); ep_viol.append(viol)
    return np.mean(ep_rewards[-50:]),np.mean(ep_energy[-50:]),np.mean(ep_viol[-50:])
results=np.array([run(seed) for seed in range(20)])
print("Q-learning mean reward, energy kWh, violation intervals:",results.mean(axis=0))
# Simple fixed thermostat baseline over 20 independent simulated days
base=[]
for seed in range(20):
    rng=np.random.default_rng(seed); temp=24.; reward=energy=viol=0.
    for t in range(48):
        occ=int(16<=t<36); price=1.8 if 28<=t<34 else .8
        action=2 if temp>24 else (1 if temp<22 and occ else 0)
        temp,r,e,v=step(temp,action,price,occ)
        reward+=r; energy+=e; viol+=bool(v>0)
    base.append([reward,energy,viol])
print("Fixed baseline mean reward, energy kWh, violation intervals:",np.mean(base,axis=0))
