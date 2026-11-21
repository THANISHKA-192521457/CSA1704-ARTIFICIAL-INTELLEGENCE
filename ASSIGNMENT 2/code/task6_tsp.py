"""TASK 6 - Delivery tour planning: own exhaustive enumeration + 2-opt hill climbing with random restarts."""
import math, random, time, json, itertools, statistics as st

def make_points(n, seed):
    rng = random.Random(seed); return [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
def dist_matrix(P): return [[math.dist(a, b) for b in P] for a in P]
def tour_len(t, D): return sum(D[t[i]][t[(i+1) % len(t)]] for i in range(len(t)))

# ------------------- own exhaustive search -------------------
# Representation: a tour is a sequence (0, c1, ..., c_{n-1}); the depot 0 is fixed first so that rotations of the same cycle
# are not generated again. Generation: depth-first extension with a 'used' flag array (each customer exactly once).
# Evaluation: running cost accumulated while extending + closing edge back to the depot at the leaf.
# Retention: best_cost / best_tour updated whenever a complete tour is strictly shorter.
def exhaustive(D, count_optimal=False):
    n = len(D); used = [False]*n; used[0] = True; path = [0]
    S = dict(best=float('inf'), tour=None, tours=0, calls=0, n_opt=0)
    def rec(last, depth, cost):
        S['calls'] += 1
        if depth == n:
            total = cost + D[last][0]; S['tours'] += 1
            if total < S['best'] - 1e-9: S['best'] = total; S['tour'] = path[:]; S['n_opt'] = 1
            elif abs(total - S['best']) <= 1e-9: S['n_opt'] += 1
            return
        row = D[last]
        for j in range(1, n):
            if not used[j]:
                used[j] = True; path.append(j)
                rec(j, depth+1, cost + row[j])
                path.pop(); used[j] = False
    rec(0, 1, 0.0)
    return S

# ------------------- local search: 2-opt hill climbing -------------------
def two_opt_descent(t, D):
    n = len(t); improved = True; evals = 0
    while improved:
        improved = False
        for i in range(n-1):
            for j in range(i+2, n if i > 0 else n-1):
                a, b, c, d = t[i], t[i+1], t[j], t[(j+1) % n]
                evals += 1
                if D[a][c] + D[b][d] < D[a][b] + D[c][d] - 1e-12:      # strictly better => reverse segment i+1..j
                    t[i+1:j+1] = reversed(t[i+1:j+1]); improved = True
    return t, evals

def hill_climb_restarts(D, restarts, rng):
    n = len(D); best = None; bl = float('inf')
    for _ in range(restarts):
        t = list(range(n)); rng.shuffle(t)
        t, _ = two_opt_descent(t, D); L = tour_len(t, D)
        if L < bl: bl, best = L, t[:]
    return best, bl

if __name__ == '__main__':
    out = {}
    # ---------- correctness check on an independently verifiable instance
    P = [(2, 3), (0, 0), (4, 3), (2, 0), (0, 3), (4, 0)]          # rectangle 4x3 with the two long-side mid-points, scrambled
    D = dist_matrix(P); S = exhaustive(D)
    brute = min(tour_len((0,)+p, D) for p in itertools.permutations(range(1, 6)))   # independent check (itertools)
    out['correctness'] = dict(points=P, perimeter_by_hand=14.0, exhaustive_best=round(S['best'], 9), itertools_best=round(brute, 9),
                              best_tour=[P[i] for i in S['tour']], tours_enumerated=S['tours'], expected_tours=math.factorial(5),
                              tours_with_optimal_length=S['n_opt'])
    print(out['correctness'])
    # ---------- growth validation (prediction: tours = (n-1)!, time ratio t(n+1)/t(n) ~ n)
    growth = []; prev = None
    for n in range(5, 13):
        D = dist_matrix(make_points(n, 100+n)); t = time.perf_counter(); S = exhaustive(D); dt = time.perf_counter()-t
        growth.append(dict(n=n, predicted_tours=math.factorial(n-1), tours_enumerated=S['tours'], recursive_calls=S['calls'],
                           time_s=round(dt, 4), time_ratio_vs_prev=round(dt/prev, 2) if prev else None, optimum=round(S['best'], 4), optimum_exact=S['best']))
        prev = dt; print(growth[-1], flush=True)
        if dt > 60: break
    out['growth'] = growth
    opt_by_n = {g['n']: g['optimum_exact'] for g in growth}
    # ---------- local search quality vs exhaustive optimum (20 runs per size per configuration)
    ls = []
    for n in (8, 9, 10, 11):
        D = dist_matrix(make_points(n, 100+n)); opt = opt_by_n[n]
        for restarts in (1, 10):
            ex, tms, hits = [], [], 0
            for run in range(20):
                rng = random.Random(run*97+n); t = time.perf_counter(); tour, L = hill_climb_restarts(D, restarts, rng); dt = time.perf_counter()-t
                ex.append(max(0.0, 100*(L-opt)/opt)); tms.append(dt); hits += abs(L-opt) < 1e-6
            ls.append(dict(n=n, restarts=restarts, mean_excess_pct=round(st.mean(ex), 3), max_excess_pct=round(max(ex), 3),
                           runs_reaching_optimum=f"{hits}/20", mean_time_ms=round(1000*st.mean(tms), 3)))
            print(ls[-1], flush=True)
    out['local_search'] = ls
    # ---------- larger n (no exhaustive optimum): compare to best tour found among 20 x 10 restarts
    big = []
    for n in (15, 20, 30, 50, 100):
        D = dist_matrix(make_points(n, 100+n)); res = []; tms = []
        for run in range(20):
            rng = random.Random(run*13+n); t = time.perf_counter(); _, L = hill_climb_restarts(D, 10, rng); tms.append(time.perf_counter()-t); res.append(L)
        bestL = min(res)
        big.append(dict(n=n, best_found=round(bestL, 3), mean_excess_over_best_pct=round(100*(st.mean(res)-bestL)/bestL, 3),
                        worst_excess_over_best_pct=round(100*(max(res)-bestL)/bestL, 3), mean_time_ms=round(1000*st.mean(tms), 2),
                        exhaustive_tours=f"{math.factorial(n-1):.3e}"))
        print(big[-1], flush=True)
    out['local_search_large'] = big
    json.dump(out, open('../results/task6_results.json', 'w'), indent=1)
