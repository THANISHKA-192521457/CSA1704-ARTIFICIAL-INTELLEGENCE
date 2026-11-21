"""TASK 5 - Constraint propagation derived from the principle:
 'a value may be removed from X's domain if some neighbour Y has no value compatible with it'.
Constraint representation: undirected conflict graph (exams sharing students); the relation on an edge is slot(X) != slot(Y).
Pseudocode of the derived procedure (also in the report):
  PROPAGATE(D, Q):                       # Q = set of directed arcs (X,Y) that must be (re)examined
    while Q not empty:
      (X,Y) = Q.pop()
      removed = {x in D[X] : no y in D[Y] with compatible(x,y)}      # unsupported values
      if removed: D[X] -= removed
         if D[X] empty: return FAIL            # domain wipe-out => no solution in this subproblem
         for every neighbour Z of X, Z != Y: Q.add((Z,X))    # only arcs whose support might have been lost
    return OK
  SOLVE(D): if PROPAGATE fails -> backtrack; if all |D|=1 -> solution; pick unassigned X (static order),
            for v in D[X]: copy D, set D'[X]={v}, PROPAGATE(D', {(Z,X)|Z in N(X)}), recurse
"""
import random, json, time, statistics as st
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Stats:
    def __init__(self): self.nodes = 0; self.backtracks = 0; self.checks = 0; self.wipeouts = 0; self.revisions = 0

def compatible(a, b): return a != b

def propagate(D, adj, Q, s, trace=None):
    Q = list(Q); inq = set(Q)
    while Q:
        X, Y = Q.pop(0); inq.discard((X, Y)); s.revisions += 1
        removed = set()
        for x in D[X]:
            supported = False
            for y in D[Y]:
                s.checks += 1
                if compatible(x, y): supported = True; break
            if not supported: removed.add(x)
        if removed:
            D[X] -= removed
            if trace is not None: trace.append((X, Y, sorted(removed), {k: len(v) for k, v in D.items()}))
            if not D[X]: s.wipeouts += 1; return False
            for Z in adj[X]:
                if Z != Y and (Z, X) not in inq: Q.append((Z, X)); inq.add((Z, X))
    return True

def all_arcs(adj): return [(x, y) for x in adj for y in adj[x]]

def solve_plain_bt(n, adj, D0, s, limit):
    assign = {}
    def rec(i):
        if i == n: return True
        if s.nodes > limit: raise TimeoutError
        for v in sorted(D0[i]):
            s.nodes += 1; ok = True
            for y in adj[i]:
                if y in assign:
                    s.checks += 1
                    if not compatible(v, assign[y]): ok = False; break
            if ok:
                assign[i] = v
                if rec(i+1): return True
                del assign[i]
        s.backtracks += 1; return False
    try: return ('SAT' if rec(0) else 'UNSAT'), dict(assign)
    except TimeoutError: return 'TIMEOUT', None

def solve_mac(n, adj, D0, s, limit, preprocess=True):
    D = {i: set(D0[i]) for i in range(n)}
    if not D[0] or any(not D[i] for i in range(n)): s.wipeouts += 1; return 'UNSAT', None
    if preprocess and not propagate(D, adj, all_arcs(adj), s): return 'UNSAT', None
    def rec(D):
        if s.nodes > limit: raise TimeoutError
        free = [i for i in range(n) if len(D[i]) > 1]
        if not free: return {i: next(iter(D[i])) for i in range(n)}
        X = free[0]
        for v in sorted(D[X]):
            s.nodes += 1
            D2 = {k: set(x) for k, x in D.items()}; D2[X] = {v}
            if propagate(D2, adj, [(Z, X) for Z in adj[X]], s):
                r = rec(D2)
                if r: return r
        s.backtracks += 1; return None
    try:
        r = rec(D); return ('SAT' if r else 'UNSAT'), r
    except TimeoutError: return 'TIMEOUT', None

def build(n, edges):
    adj = {i: set() for i in range(n)}
    for a, b in edges: adj[a].add(b); adj[b].add(a)
    return adj

def valid(sol, adj):
    return sol is not None and all(sol[a] != sol[b] for a in adj for b in adj[a])

def random_instance(n, avg_deg, slots, seed):
    rng = random.Random(seed); m = int(n*avg_deg/2); edges = set()
    while len(edges) < m:
        a, b = rng.sample(range(n), 2); edges.add((min(a, b), max(a, b)))
    return build(n, edges), {i: set(range(1, slots+1)) for i in range(n)}

def heat(ax, D, n, slots, title, names):
    M = [[1 if (j+1) in D[i] else 0 for j in range(slots)] for i in range(n)]
    ax.imshow(M, cmap='Blues', vmin=0, vmax=1.4, aspect='auto')
    ax.set_xticks(range(slots)); ax.set_xticklabels([f'S{j+1}' for j in range(slots)])
    ax.set_yticks(range(n)); ax.set_yticklabels(names); ax.set_title(title, fontsize=10)
    for i in range(n):
        for j in range(slots): ax.text(j, i, '✓' if M[i][j] else '', ha='center', va='center', color='white', fontsize=9)

if __name__ == '__main__':
    out = {}; LIM = 300000
    # ---------- Instance 1a: propagation alone solves
    names = list('ABCDEFG')
    edges = [(0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(3,6),(5,6)]
    adj = build(7, edges); D = {0: {1}, 1: {1,2}, 2: {1,2,3}, 3: {1,2,3}, 4: {1,2,3}, 5: {3}, 6: {1,2,3}}
    INIT1A = {k: set(v) for k, v in D.items()}; s = Stats(); trace = []
    ok = propagate(D, adj, all_arcs(adj), s, trace)
    sol = {k: next(iter(v)) for k, v in D.items()}
    out['instance_1a'] = dict(result='solved by propagation only', ok=ok, final={names[k]: sorted(v) for k, v in D.items()},
                              valid=valid(sol, adj), checks=s.checks, revisions=s.revisions, wipeouts=s.wipeouts,
                              search_nodes=0, trace=[(names[x], names[y], r) for x, y, r, _ in trace])
    print('1a', out['instance_1a'])
    # ---------- Instance 1b: failure detected without search
    adj_b = build(3, [(0,1),(0,2),(1,2)]); Db = {0: {1}, 1: {1,2}, 2: {1,2}}; s = Stats(); tr = []
    ok_b = propagate(Db, adj_b, all_arcs(adj_b), s, tr)
    out['instance_1b'] = dict(result='no solution detected by propagation only', ok=ok_b, checks=s.checks, wipeouts=s.wipeouts,
                              search_nodes=0, trace=[(x, y, r) for x, y, r, _ in tr])
    print('1b', out['instance_1b'])
    # ---------- Instance 2: challenge (K5, 4 slots: AC cannot see the pigeon-hole contradiction)
    k5 = build(5, [(i, j) for i in range(5) for j in range(i+1, 5)]); Dk = {i: {1,2,3,4} for i in range(5)}
    s0 = Stats(); D = {k: set(v) for k, v in Dk.items()}; ac_ok = propagate(D, k5, all_arcs(k5), s0)
    res2 = dict(ac3_alone_detects_failure=not ac_ok, domains_after_ac3={k: sorted(v) for k, v in D.items()}, ac3_checks=s0.checks)
    sp, sm = Stats(), Stats()
    r1 = solve_plain_bt(5, k5, Dk, sp, LIM)[0]; r2 = solve_mac(5, k5, Dk, sm, LIM)[0]
    res2['plain_bt'] = dict(result=r1, nodes=sp.nodes, backtracks=sp.backtracks, checks=sp.checks, wipeouts=0)
    res2['mac'] = dict(result=r2, nodes=sm.nodes, backtracks=sm.backtracks, checks=sm.checks, wipeouts=sm.wipeouts)
    out['instance_2_K5_4slots'] = res2; print('2', res2)
    # ---------- scaling on random timetables (3 slots, mean conflicts per exam = 4.2, 10 seeds)
    scal = []
    for n in (10, 15, 20, 25, 30, 40, 50):
        agg = {'bt': [], 'mac': []}; outcomes = {'bt': [], 'mac': []}
        for seed in range(10):
            adj, D0 = random_instance(n, 4.2, 3, 1000*n+seed)
            for key, fn in (('bt', lambda s_: solve_plain_bt(n, adj, D0, s_, LIM)), ('mac', lambda s_: solve_mac(n, adj, D0, s_, LIM))):
                s = Stats(); t = time.perf_counter(); res, sol = fn(s); dt = time.perf_counter()-t
                if res == 'SAT': assert valid(sol, adj)
                agg[key].append((s.nodes, s.backtracks, s.checks, s.wipeouts, dt)); outcomes[key].append(res)
        row = dict(n=n)
        for key in agg:
            for i, nm in enumerate(('nodes', 'backtracks', 'checks', 'wipeouts', 'time_s')):
                row[f'{key}_{nm}'] = round(st.mean(a[i] for a in agg[key]), 3 if nm == 'time_s' else 1)
            row[f'{key}_timeouts'] = outcomes[key].count('TIMEOUT')
        row['sat'] = outcomes['mac'].count('SAT'); row['unsat'] = outcomes['mac'].count('UNSAT')
        scal.append(row); print(row, flush=True)
    out['scaling'] = scal
    json.dump(out, open('../results/task5_results.json', 'w'), indent=1)
    # ---------- figures
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    heat(axs[0], INIT1A, 7, 3, 'Initial domains (unary restrictions)', names)
    Dmid = {k: set(v) for k, v in D0.items()}; s = Stats(); tr2 = []
    heat(axs[1], {k: set(v) for k, v in out['instance_1a']['final'].items() and {i: set(out['instance_1a']['final'][names[i]]) for i in range(7)}.items()}, 7, 3, 'After AC propagation (solved)', names)
    sizes = [sum(len(v) for v in INIT1A.values())] + [sum(t[3].values()) for t in trace]
    axs[2].step(range(len(sizes)), sizes, where='post', marker='o'); axs[2].set_xlabel('value-removal events'); axs[2].set_ylabel('total domain size')
    axs[2].set_title('Shrinking of domains', fontsize=10); axs[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig('../figures/fig5a_domains.png', dpi=170); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4)); ns = [r['n'] for r in scal]
    ax[0].semilogy(ns, [max(r['bt_nodes'], 1) for r in scal], 'o-', label='backtracking only'); ax[0].semilogy(ns, [max(r['mac_nodes'], 1) for r in scal], 's-', label='backtracking + propagation')
    ax[0].set_xlabel('number of exams'); ax[0].set_ylabel('mean nodes visited'); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].semilogy(ns, [max(r['bt_checks'], 1) for r in scal], 'o-', label='backtracking only'); ax[1].semilogy(ns, [max(r['mac_checks'], 1) for r in scal], 's-', label='backtracking + propagation')
    ax[1].set_xlabel('number of exams'); ax[1].set_ylabel('mean constraint checks'); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig('../figures/fig5b_scaling.png', dpi=170); plt.close()
