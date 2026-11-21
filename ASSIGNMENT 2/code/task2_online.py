"""TASK 2 - Online DFS agent and LRTA* for a rover in an unmapped facility (grid mazes)."""
import random, json, statistics as st
from collections import deque
DIRS = [(-1,0),(1,0),(0,-1),(0,1)]

def make_maze(n, braid=0.0, seed=0):
    """n odd. Perfect maze by randomised DFS, then 'braid' = prob. of removing each dead end."""
    rng = random.Random(seed)
    free = set()
    def carve(start):
        stack = [start]; free.add(start)
        while stack:
            r, c = stack[-1]
            cand = [(r+dr*2, c+dc*2, dr, dc) for dr, dc in DIRS
                    if 0 < r+dr*2 < n-1 and 0 < c+dc*2 < n-1 and (r+dr*2, c+dc*2) not in free]
            if cand:
                nr, nc, dr, dc = rng.choice(cand)
                free.add((r+dr, c+dc)); free.add((nr, nc)); stack.append((nr, nc))
            else: stack.pop()
    carve((1, 1))
    if braid > 0:
        for cell in sorted(free):
            if len(nbrs(free, cell)) == 1 and rng.random() < braid:
                r, c = cell
                walls = [(r+dr, c+dc) for dr, dc in DIRS
                         if 0 < r+dr < n-1 and 0 < c+dc < n-1 and (r+dr, c+dc) not in free
                         and (r+2*dr, c+2*dc) in free]
                if walls: free.add(rng.choice(walls))
    return free

def nbrs(free, p):
    return [(p[0]+dr, p[1]+dc) for dr, dc in DIRS if (p[0]+dr, p[1]+dc) in free]

def bfs_dist(free, s):
    d = {s: 0}; q = deque([s])
    while q:
        x = q.popleft()
        for m in nbrs(free, x):
            if m not in d: d[m] = d[x]+1; q.append(m)
    return d

def dead_end_fraction(free):
    return sum(len(nbrs(free, p)) == 1 for p in free) / len(free)

def online_dfs(free, start, target, rng=None):
    """Agent only senses the free neighbours of its current cell (learned on arrival)."""
    pos = start; visited = {start}; untried = {start: nbrs(free, start)}
    if rng: rng.shuffle(untried[start])
    stack = []; moves = 0
    while pos != target:
        untried[pos] = [m for m in untried[pos] if m not in visited]
        if untried[pos]:
            nxt = untried[pos].pop(0); stack.append(pos)
        elif stack:
            nxt = stack.pop()
        else:
            return dict(success=False, moves=moves, unique=len(visited), revisits=moves-(len(visited)-1))
        moves += 1; pos = nxt
        if pos not in visited:
            visited.add(pos); untried[pos] = nbrs(free, pos)
            if rng: rng.shuffle(untried[pos])
    return dict(success=True, moves=moves, unique=len(visited), revisits=moves-(len(visited)-1))

def lrta_trial(free, start, target, H, rng, cap):
    """One LRTA* trial; H persists across trials (learning). Unit step cost; h0 = Manhattan to target."""
    h0 = lambda p: abs(p[0]-target[0]) + abs(p[1]-target[1])
    pos = start; moves = 0; visited = {start}
    while pos != target and moves < cap:
        ns = nbrs(free, pos)
        vals = [(1 + H.get(m, h0(m)), m) for m in ns]
        best = min(v for v, _ in vals)
        nxt = rng.choice([m for v, m in vals if v == best])
        H[pos] = max(H.get(pos, h0(pos)), best)      # LRTA* update: H(s) <- min_a [c + H(s')]
        pos = nxt; moves += 1; visited.add(pos)
    return dict(success=pos == target, moves=moves, unique=len(visited))

def pick_target(free, start, frac):
    d = bfs_dist(free, start); mx = max(d.values()); want = max(1, round(frac*mx))
    return min(d, key=lambda p: (abs(d[p]-want), p)), d

def isolated_target(free, n):
    """a free cell completely surrounded by walls -> unreachable"""
    for r in range(2, n-2, 2):
        for c in range(2, n-2, 2):
            if (r, c) not in free and all((r+dr, c+dc) not in free for dr, dc in DIRS):
                return (r, c)

N_MAZES = 20
out = {}
def mean(x): return round(st.mean(x), 2)

# E1: target distance (n=31, perfect maze)
rows = []
for frac in (0.25, 0.5, 0.75, 1.0):
    mv, rv, cr, opt, lr1, lr10 = [], [], [], [], [], []
    for sd in range(N_MAZES):
        free = make_maze(31, 0.0, sd); tgt, d = pick_target(free, (1, 1), frac)
        r = online_dfs(free, (1, 1), tgt, random.Random(sd))
        mv.append(r['moves']); rv.append(r['revisits']); cr.append(r['moves']/d[tgt]); opt.append(d[tgt])
        H = {}; rg = random.Random(sd); tr = [lrta_trial(free, (1,1), tgt, H, rg, 50000)['moves'] for _ in range(10)]
        lr1.append(tr[0]/d[tgt]); lr10.append(tr[9]/d[tgt])
    rows.append(dict(target_frac=frac, optimal_cost=mean(opt), dfs_moves=mean(mv), dfs_revisits=mean(rv),
                     dfs_competitive_ratio=mean(cr), lrta_trial1_ratio=mean(lr1), lrta_trial10_ratio=mean(lr10)))
out['E1_distance'] = rows

# E2: unreachable targets
rows = []
for n in (11, 21, 31, 41):
    mv, cells, lrmv = [], [], []
    for sd in range(N_MAZES):
        free = make_maze(n, 0.0, sd); tgt = isolated_target(free | set(), n)
        if tgt is None: continue
        free2 = free | {tgt}
        r = online_dfs(free2, (1, 1), tgt, random.Random(sd))
        assert not r['success']; mv.append(r['moves']); cells.append(len(free))
        rg = random.Random(sd); lrmv.append(lrta_trial(free2, (1,1), tgt, {}, rg, 20*len(free))['moves'])
    rows.append(dict(size=n, free_cells=mean(cells), dfs_moves_until_giving_up=mean(mv),
                     dfs_declares_failure=True, moves_per_cell=round(st.mean(m/c for m, c in zip(mv, cells)), 2),
                     lrta_moves_in_cap_20x_cells=mean(lrmv), lrta_terminates=False))
out['E2_unreachable'] = rows

# E3: maze size (farthest target, perfect maze)
rows = []
for n in (11, 21, 31, 41, 51):
    mv, cr, opt = [], [], []
    for sd in range(N_MAZES):
        free = make_maze(n, 0.0, sd); tgt, d = pick_target(free, (1, 1), 1.0)
        r = online_dfs(free, (1, 1), tgt, random.Random(sd)); mv.append(r['moves']); cr.append(r['moves']/d[tgt]); opt.append(d[tgt])
    rows.append(dict(size=n, free_cells=len(free), optimal_cost=mean(opt), dfs_moves=mean(mv), competitive_ratio=mean(cr),
                     worst_ratio=round(max(cr), 2)))
out['E3_size'] = rows

# E4: dead-end density (n=31, mid-far target frac 0.75)
rows = []
for braid in (0.0, 0.25, 0.5, 0.75, 1.0):
    de, mv, rv, cr = [], [], [], []
    for sd in range(N_MAZES):
        free = make_maze(31, braid, sd); tgt, d = pick_target(free, (1, 1), 0.75)
        r = online_dfs(free, (1, 1), tgt, random.Random(sd))
        de.append(dead_end_fraction(free)); mv.append(r['moves']); rv.append(r['revisits']); cr.append(r['moves']/d[tgt])
    rows.append(dict(braid=braid, dead_end_pct=round(100*st.mean(de), 1), dfs_moves=mean(mv), dfs_revisits=mean(rv),
                     competitive_ratio=mean(cr)))
out['E4_dead_ends'] = rows

# E5: repeated missions in the SAME facility (changed condition) n=31, farthest target, 20 mazes, 30 missions
K = 30; cum = {k: [0]*K for k in ('dfs_every_time', 'lrta_learning', 'map_once_then_astar')}
per_trial = []
for sd in range(N_MAZES):
    free = make_maze(31, 0.25, sd); tgt, d = pick_target(free, (1, 1), 1.0); opt = d[tgt]
    dfs = online_dfs(free, (1, 1), tgt, random.Random(sd))['moves']
    H = {}; rg = random.Random(sd); lt = [lrta_trial(free, (1,1), tgt, H, rg, 50000)['moves'] for _ in range(K)]
    c1 = c2 = c3 = 0
    for k in range(K):
        c1 += dfs; c2 += lt[k]; c3 += dfs if k == 0 else opt
        cum['dfs_every_time'][k] += c1/N_MAZES; cum['lrta_learning'][k] += c2/N_MAZES; cum['map_once_then_astar'][k] += c3/N_MAZES
    per_trial.append([x/opt for x in lt])
out['E5_missions'] = dict(cumulative_moves={k: [round(x, 1) for x in v] for k, v in cum.items()},
                         lrta_ratio_per_trial=[round(st.mean(t[k] for t in per_trial), 2) for k in range(K)])
json.dump(out, open('../results/task2_results.json', 'w'), indent=1)
for k, v in out.items():
    print(k)
    if isinstance(v, list):
        for r in v: print('  ', r)
    else:
        print('  cumulative @1,5,10,30:', {a: [b[0], b[4], b[9], b[29]] for a, b in v['cumulative_moves'].items()})
        print('  LRTA ratio per trial:', v['lrta_ratio_per_trial'])
