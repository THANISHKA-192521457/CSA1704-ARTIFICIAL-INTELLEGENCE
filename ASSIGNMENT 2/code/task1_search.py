"""TASK 1 - Uniform-Cost, Greedy Best-First and A* on campus-shuttle grid maps.
Grid legend: '#' obstacle, '.' normal road (cost 1), '~' rough/mud road (cost 5).
Heuristic (own design): h(n) = c_min * Manhattan(n, goal).
Admissibility (relaxed problem): remove all obstacles and let every cell cost c_min;
the exact cost in the relaxed problem is c_min*Manhattan, and relaxing can only lower
the optimum, so h <= h*.   Consistent as well (a step changes h by at most c_min <= step cost).
"""
import heapq, random, time, json, itertools
from collections import deque

COST = {'.': 1, '~': 5, 'S': 1, 'G': 1}
CMIN = 1

def manhattan(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
def h(n, goal): return CMIN * manhattan(n, goal)

def neighbours(grid, p):
    R, C = len(grid), len(grid[0])
    r, c = p
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        nr, nc = r+dr, c+dc
        if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] != '#':
            yield (nr, nc), COST[grid[nr][nc]]

def search(grid, start, goal, mode):
    """mode in {'UCS','GBFS','ASTAR'}; graph search with best-g bookkeeping."""
    t0 = time.perf_counter()
    def f(g, n):
        return g if mode == 'UCS' else (h(n, goal) if mode == 'GBFS' else g + h(n, goal))
    tie = itertools.count()
    best_g = {start: 0}; parent = {start: None}
    open_heap = [(f(0, start), next(tie), start)]
    in_frontier = {start}; closed = set()
    expanded = generated = 1 - 1; generated = 1; max_front = 1
    while open_heap:
        _, _, n = heapq.heappop(open_heap)
        if n in closed: continue
        in_frontier.discard(n)
        if n == goal:
            path = []; x = n
            while x is not None: path.append(x); x = parent[x]
            return dict(found=True, cost=best_g[n], expanded=expanded, generated=generated,
                        max_frontier=max_front, length=len(path)-1, time=time.perf_counter()-t0)
        closed.add(n); expanded += 1
        for m, c in neighbours(grid, n):
            g2 = best_g[n] + c
            if m in closed and not (mode != 'GBFS' and g2 < best_g.get(m, 1e18)):
                continue
            if m not in best_g or (mode != 'GBFS' and g2 < best_g[m]):
                best_g[m] = g2; parent[m] = n
                heapq.heappush(open_heap, (f(g2, m), next(tie), m))
                in_frontier.add(m); generated += 1; closed.discard(m)
                max_front = max(max_front, len(in_frontier))
    return dict(found=False, cost=None, expanded=expanded, generated=generated,
                max_frontier=max_front, length=None, time=time.perf_counter()-t0)

def reachable(grid, s, g):
    seen = {s}; q = deque([s])
    while q:
        n = q.popleft()
        if n == g: return True
        for m, _ in neighbours(grid, n):
            if m not in seen: seen.add(m); q.append(m)
    return False

# ---------------- map builders ----------------
def blank(R, C): return [['.']*C for _ in range(R)]
def to_str(g): return [''.join(r) for r in g]

def map_open(R=30, C=30, density=0.05, seed=1):
    rng = random.Random(seed)
    while True:
        g = blank(R, C)
        for r in range(R):
            for c in range(C):
                if rng.random() < density: g[r][c] = '#'
        g[0][0] = g[R-1][C-1] = '.'
        if reachable(g, (0,0), (R-1,C-1)): return to_str(g), (0,0), (R-1,C-1)

def map_cluttered(R=30, C=30, density=0.30, seed=7):
    return map_open(R, C, density, seed)

def map_maze(n=31, seed=3):
    rng = random.Random(seed)
    g = [['#']*n for _ in range(n)]
    def carve(r, c):
        g[r][c] = '.'
        d = [(0,2),(0,-2),(2,0),(-2,0)]; rng.shuffle(d)
        for dr, dc in d:
            nr, nc = r+dr, c+dc
            if 0 < nr < n-1 and 0 < nc < n-1 and g[nr][nc] == '#':
                g[r+dr//2][c+dc//2] = '.'; carve(nr, nc)
    import sys; sys.setrecursionlimit(10000)
    carve(1, 1)
    return to_str(g), (1,1), (n-2,n-2)

def map_utrap(R=30, C=30):
    g = blank(R, C)
    for c in range(6, 15): g[8][c] = g[22][c] = '#'       # cup top and bottom
    for r in range(8, 23): g[r][14] = '#'                  # cup back wall (faces the goal)
    return to_str(g), (15, 10), (15, 28)                   # start inside the cup, opening faces left

def map_mud(R=30, C=30):
    g = blank(R, C)
    for r in range(10, 20):
        for c in range(C): g[r][c] = '~'
    for r in range(10, 20): g[r][2] = '.'                  # a cheap corridor exists at column 2
    return to_str(g), (2, 6), (27, 27)

MAPS = {
 'A_open_5pct': map_open(),
 'B_cluttered_30pct': map_cluttered(),
 'C_maze': map_maze(),
 'D_dead_end_trap': map_utrap(),
 'E_mud_band_edge_case': map_mud(),
}

if __name__ == '__main__':
    rows = []
    print("PREDICTIONS (written before running) are in the report, Section 1.2")
    for name, (grid, s, g) in MAPS.items():
        res = {m: search(grid, s, g, m) for m in ('UCS', 'GBFS', 'ASTAR')}
        opt = res['UCS']['cost']
        for m, r in res.items():
            r = dict(r); r['map'] = name; r['algo'] = m
            r['optimal'] = (r['cost'] == opt); r['time_ms'] = round(r.pop('time')*1000, 2)
            rows.append(r)
            print(f"{name:22s} {m:6s} cost={r['cost']} opt={r['optimal']} exp={r['expanded']} "
                  f"gen={r['generated']} maxF={r['max_frontier']} t={r['time_ms']}ms")
    # open-road scenario (95% of trips): 40 random trips on 5% obstacle map
    grid, _, _ = MAPS['A_open_5pct']; rng = random.Random(11)
    free = [(r, c) for r in range(30) for c in range(30) if grid[r][c] == '.']
    agg = {m: dict(exp=0, opt=0, n=0, extra=0) for m in ('UCS','GBFS','ASTAR')}
    trips = 0
    while trips < 200:
        s, g = rng.sample(free, 2)
        if not reachable(grid, s, g): continue
        trips += 1
        res = {m: search(grid, s, g, m) for m in agg}
        for m in agg:
            agg[m]['exp'] += res[m]['expanded']; agg[m]['n'] += 1
            agg[m]['opt'] += res[m]['cost'] == res['UCS']['cost']
            agg[m]['extra'] += res[m]['cost'] / res['UCS']['cost'] - 1
    trips_summary = {m: dict(mean_expanded=round(a['exp']/a['n'], 1), pct_optimal=round(100*a['opt']/a['n'], 1),
                             mean_cost_excess_pct=round(100*a['extra']/a['n'], 2)) for m, a in agg.items()}
    print("200 random trips on map A:", trips_summary)
    json.dump(dict(rows=rows, trips=trips_summary, maps={k: v[0] for k, v in MAPS.items()}),
              open('../results/task1_results.json', 'w'), indent=1)
