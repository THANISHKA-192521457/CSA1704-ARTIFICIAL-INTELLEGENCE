"""Task 12: localisation lost -> search in belief space (grid-corridor hospital map)."""
import random, itertools
from collections import deque

W = H = 6
ACTS = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
DOCK = (0, 0)

def make_map(seed, density=0.12):
    rng = random.Random(seed)
    while True:
        walls = {(r, c) for r in range(H) for c in range(W)
                 if (r, c) != DOCK and rng.random() < density}
        free = {(r, c) for r in range(H) for c in range(W)} - walls
        # connectivity check from dock
        seen = {DOCK}; q = deque([DOCK])
        while q:
            r, c = q.popleft()
            for dr, dc in ACTS.values():
                n = (r + dr, c + dc)
                if n in free and n not in seen: seen.add(n); q.append(n)
        if seen == free: return free

def step(free, p, a):
    dr, dc = ACTS[a]; n = (p[0] + dr, p[1] + dc)
    return n if n in free else p                      # bump into wall = stay

def percept(free, p):                                 # local sensor: which sides are blocked
    return tuple((p[0] + dr, p[1] + dc) not in free for dr, dc in ACTS.values())

def apply(free, B, a):
    return frozenset(step(free, p, a) for p in B)

# ---------- (A) no sensor : BFS over belief states (sensorless problem)
def solve_blind(free, B0, cap=400000):
    start = frozenset(B0); gen = 1
    parent = {start: None}; q = deque([start])
    while q:
        B = q.popleft()
        if gen > cap: return None, gen          # cap reached: report as '>cap'
        if B <= {DOCK}:
            plan = []
            while parent[B]: B, a = parent[B]; plan.append(a)
            return len(plan), gen
        for a in ACTS:
            C = apply(free, B, a)
            if C not in parent:
                parent[C] = (B, a); gen += 1; q.append(C)
    return None, gen

# ---------- (B) local sensor : conditional plan by AND-OR search, iterative deepening
def split(free, B):
    g = {}
    for p in B: g.setdefault(percept(free, p), set()).add(p)
    return [frozenset(v) for v in g.values()]

def solve_sensor(free, B0, maxlen=60):
    gen = [1]
    def andor(B, lim, memo, bad):
        if B <= {DOCK}: return 0
        if lim == 0: return None
        key = (B, lim)
        if key in memo: return memo[key]
        best = None
        for a in ACTS:
            kids = split(free, apply(free, B, a)); gen[0] += len(kids)
            worst = 0
            for k in kids:
                if k == B and True:        # no progress branch with identical belief: skip
                    worst = None; break
                r = andor(k, lim - 1, memo, bad)
                if r is None: worst = None; break
                worst = max(worst, r)
            if worst is not None and (best is None or 1 + worst < best):
                best = 1 + worst
                if best == 1: break
        memo[key] = best
        return best
    init = split(free, set(B0))                       # first percept splits the belief
    gen[0] += len(init)
    total = 0
    for B in init:                                    # plan length = worst branch
        for lim in range(0, maxlen + 1):
            r = andor(B, lim, {}, None)
            if r is not None: total = max(total, r); break
        else:
            return None, gen[0]
    return total, gen[0]

if __name__ == "__main__":
    SEED = 3
    free = make_map(SEED)
    print(f"map seed={SEED}: {len(free)} free junctions of {W*H}; dock={DOCK}")
    for r in range(H):
        print("   " + "".join("D" if (r, c) == DOCK else ("." if (r, c) in free else "#") for c in range(W)))
    print()
    print(f"{'k':>2} {'trial seed':>10} | {'no-sensor: beliefs gen':>22} {'plan len':>8} | {'sensor: beliefs gen':>19} {'worst-case plan':>15}")
    agg = {}
    for k in (2, 4, 8):
        for ts in (1, 2, 3):
            rng = random.Random(100 * k + ts)
            B0 = set(rng.sample(sorted(free), k))
            pl0, g0 = solve_blind(free, B0)
            pl1, g1 = solve_sensor(free, B0)
            s0 = str(pl0) if pl0 is not None else 'cap'
            print(f"{k:>2} {ts:>10} | {g0:>22} {s0:>8} | {g1:>19} {str(pl1):>15}")
            agg.setdefault(k, []).append((g0, pl0, g1, pl1))
    print("\nAverages over 3 trials")
    for k, v in agg.items():
        m = lambda i: sum(x[i] for x in v if x[i] is not None) / max(1, sum(1 for x in v if x[i] is not None))
        print(f"k={k}: no-sensor gen={m(0):.0f} plan={m(1):.1f} | sensor gen={m(2):.0f} plan={m(3):.1f}")
