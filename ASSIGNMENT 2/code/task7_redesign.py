"""TASK 7 - Diagnose, redesign, validate.  Chosen solution: exhaustive tour enumeration of Task 6.
Bottleneck (from Task 6 evidence): calls = ~2.7 * (n-1)!, every one of the (n-1)! complete tours is built and priced even when
its first few edges already make it longer than a tour we hold.  Redesign (own derivation, no named textbook algorithm assumed):
  R1 prefix cost cut      : abandon a partial tour when cost_so_far >= best_known
  R2 optimistic completion: abandon when cost_so_far + (cheapest possible cost of the edges still to be added) >= best_known
                            cheapest possible = min_out[last] + sum_{u unvisited} min_out[u]   (each remaining edge leaves exactly one such city)
  R3 child ordering       : extend to nearest unvisited city first, so good tours are found early and best_known tightens quickly
  R4 reuse                : seed best_known with the 2-opt local-search tour already produced in Task 6 (information re-use)
Every cut removes only tours whose length is provably >= best_known, so the optimum is never lost."""
import random, time, json, math, statistics as st
from task6_tsp import make_points, dist_matrix, tour_len, exhaustive, hill_climb_restarts

def bnb(D, use_lb=True, order=True, ub=float('inf'), call_cap=None):
    n = len(D); min_out = [min(D[i][j] for j in range(n) if j != i) for i in range(n)]
    used = [False]*n; used[0] = True; path = [0]
    S = dict(best=ub, tour=None, calls=0, tours=0, prunes=0, capped=False)
    def rec(last, depth, cost, rem):
        S['calls'] += 1
        if call_cap and S['calls'] > call_cap: S['capped'] = True; raise TimeoutError
        if depth == n:
            total = cost + D[last][0]; S['tours'] += 1
            if total < S['best'] - 1e-12: S['best'] = total; S['tour'] = path[:]
            return
        cand = [j for j in range(1, n) if not used[j]]
        if order: cand.sort(key=lambda j: D[last][j])
        for j in cand:
            c2 = cost + D[last][j]
            rem2 = rem - min_out[j]
            bound = c2 + (min_out[j] + rem2 if use_lb else 0.0)
            if bound >= S['best'] - 1e-12: S['prunes'] += 1; continue
            used[j] = True; path.append(j)
            rec(j, depth+1, c2, rem2)
            path.pop(); used[j] = False
    try: rec(0, 1, 0.0, sum(min_out[1:]))
    except TimeoutError: pass
    return S

if __name__ == '__main__':
    out = {}
    # ---------- ablation + comparison on identical instances (same seeds as Task 6)
    rows = []
    for n in (8, 9, 10, 11, 12):
        D = dist_matrix(make_points(n, 100+n))
        if n <= 11:
            t = time.perf_counter(); E = exhaustive(D); te = time.perf_counter()-t; calls_e, opt = E['calls'], E['best']
        else:
            calls_e, opt, te = 108505112, 286.12777922891433, 36.6   # measured in task6 (full run, same instance)
        row = dict(n=n, exhaustive_calls=calls_e, exhaustive_time_s=round(te, 3), optimum=round(opt, 4))
        for name, kw in (('R1_prefix_only', dict(use_lb=False, order=False)), ('R1R2_bound', dict(use_lb=True, order=False)),
                         ('R1R2R3_bound_order', dict(use_lb=True, order=True))):
            t = time.perf_counter(); S = bnb(D, call_cap=6_000_000, **kw); dt = time.perf_counter()-t
            row[name] = dict(calls=S['calls'], tours=S['tours'], prunes=S['prunes'], time_s=round(dt, 4), optimal=abs(S['best']-opt) < 1e-9)
        t = time.perf_counter(); tour, L = hill_climb_restarts(D, 10, random.Random(5)); t_ls = time.perf_counter()-t
        t = time.perf_counter(); S = bnb(D, use_lb=True, order=True, ub=L + 1e-9); dt = time.perf_counter()-t
        row['R1R2R3R4_full'] = dict(calls=S['calls'], tours=S['tours'], prunes=S['prunes'], time_s=round(dt + t_ls, 4),
                                    seed_ls_excess_pct=round(100*(L-opt)/opt, 3), optimal=abs(min(S['best'], L)-opt) < 1e-9)
        row['reduction_pct_full_vs_exhaustive'] = round(100*(1 - row['R1R2R3R4_full']['calls']/calls_e), 3)
        rows.append(row); print(json.dumps(row), flush=True)
    out['comparison'] = rows
    # ---------- correctness over many random instances (n=9, 30 instances) : redesigned == exhaustive
    agree = 0; ratios = []
    for sd in range(30):
        D = dist_matrix(make_points(9, 5000+sd)); E = exhaustive(D); tour, L = hill_climb_restarts(D, 10, random.Random(sd))
        S = bnb(D, ub=L + 1e-9); best = min(S['best'], L); agree += abs(best-E['best']) < 1e-9; ratios.append(S['calls']/E['calls'])
    out['correctness_30_instances_n9'] = dict(agree=agree, total=30, mean_call_fraction=round(st.mean(ratios), 5))
    print(out['correctness_30_instances_n9'])
    # ---------- scalability of the redesign alone (no exhaustive possible): mean of 5 instances
    sc = []
    for n in (12, 14, 16):
        cs, ts, oks = [], [], 0
        for sd in range(5):
            D = dist_matrix(make_points(n, 9000+n*10+sd)); t = time.perf_counter()
            tour, L = hill_climb_restarts(D, 10, random.Random(sd)); S = bnb(D, ub=L + 1e-9, call_cap=3_000_000)
            ts.append(time.perf_counter()-t); cs.append(S['calls']); oks += not S['capped']
        sc.append(dict(n=n, mean_calls=round(st.mean(cs)), mean_time_s=round(st.mean(ts), 3), finished=f"{oks}/5", exhaustive_tours=f"{math.factorial(n-1):.2e}"))
        print(sc[-1], flush=True)
    out['scalability'] = sc
    json.dump(out, open('../results/task7_results.json', 'w'), indent=1)
