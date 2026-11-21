"""Hospital medicine-delivery robot: uninformed search (CO1 / Assignment 1)."""
import random, time, heapq, sys
from collections import deque

# ------------------------------------------------------------------ graph generator
def build_hospital(n, seed, island=8, window=1000, p_local=0.9):
    """Connected hospital graph on n-island nodes + a sealed 'island' ward block.
    Edge encoded as (v<<4)|w, w = travel time 1..9.  Pharmacy = node 0."""
    rng = random.Random(seed)
    m = n - island
    adj = [[] for _ in range(n)]
    def pick(i):
        if rng.random() < p_local:
            return rng.randrange(max(0, i - window), i)
        return rng.randrange(0, i)                      # lift / bridge shortcut
    def add(u, v):
        if u == v:
            return
        w = rng.randint(1, 9)
        adj[u].append((v << 4) | w); adj[v].append((u << 4) | w)
    for i in range(1, m):                               # spanning tree => connected
        add(i, pick(i))
    for _ in range(m // 2):                             # extra corridors => cycles
        i = rng.randrange(1, m); add(i, pick(i))
    for i in range(m + 1, n):                           # sealed ward block (unreachable)
        add(i, rng.randrange(m, i))
    return adj

# ------------------------------------------------------------------ algorithms
# every algorithm returns (path, cost, stats) ; stats = dict(gen, exp, maxf)
def path_cost(adj, path):
    c = 0
    for a, b in zip(path, path[1:]):
        c += min(e & 15 for e in adj[a] if e >> 4 == b)
    return c

def rebuild(parent, g):
    p = [g]
    while parent[p[-1]] != -1:
        p.append(parent[p[-1]])
    return p[::-1]

def bfs(adj, s, g):
    gen = exp = 1; maxf = 1
    if s == g:
        return [s], 0, dict(gen=1, exp=0, maxf=1)
    parent = {s: -1}; q = deque([s])
    while q:
        u = q.popleft(); exp += 1
        for e in adj[u]:
            v = e >> 4
            if v not in parent:                           # explored/frontier check
                parent[v] = u; gen += 1
                if v == g:                                # early goal test
                    p = rebuild(parent, g)
                    return p, path_cost(adj, p), dict(gen=gen, exp=exp, maxf=max(maxf, len(q) + 1))
                q.append(v)
        if len(q) > maxf: maxf = len(q)
    return None, None, dict(gen=gen, exp=exp, maxf=maxf)

def ucs(adj, s, g):
    gen = 1; exp = 0
    best = {s: 0}; parent = {s: -1}; pq = [(0, s)]; maxf = 1; done = set()
    while pq:
        c, u = heapq.heappop(pq)
        if u in done: continue                            # stale entry
        done.add(u); exp += 1
        if u == g:                                        # goal test on expansion
            return rebuild(parent, g), c, dict(gen=gen, exp=exp, maxf=maxf)
        for e in adj[u]:
            v = e >> 4; nc = c + (e & 15)
            if v not in done and nc < best.get(v, 1 << 60):
                best[v] = nc; parent[v] = u; gen += 1
                heapq.heappush(pq, (nc, v))
        if len(pq) > maxf: maxf = len(pq)
    return None, None, dict(gen=gen, exp=exp, maxf=maxf)

def dfs(adj, s, g):
    gen = 1; exp = 0; maxf = 1
    parent = {s: -1}; st = [s]
    while st:
        u = st.pop(); exp += 1
        if u == g:
            p = rebuild(parent, g)
            return p, path_cost(adj, p), dict(gen=gen, exp=exp, maxf=maxf)
        for e in adj[u]:
            v = e >> 4
            if v not in parent:
                parent[v] = u; gen += 1; st.append(v)
        if len(st) > maxf: maxf = len(st)
    return None, None, dict(gen=gen, exp=exp, maxf=maxf)

def iddfs(adj, s, g):
    """Recursive depth-limited DFS repeated for limit = 0,1,2,... ; depth-aware
    duplicate detection (revisit a node only if reached at a smaller depth)."""
    sys.setrecursionlimit(10000)
    tot = dict(gen=1, exp=0, maxf=1)
    for limit in range(0, 10**9):
        seen = {s: 0}; pruned = set(); stats = dict(gen=0, exp=0, depth=0, maxd=0)
        def dls(u, d, limit):
            stats['exp'] += 1
            if d > stats['maxd']: stats['maxd'] = d       # recursion stack depth = frontier proxy
            if u == g:
                return [u]
            if d == limit:                                # frontier of the depth limit
                for e in adj[u]:
                    v = e >> 4
                    if v not in seen: pruned.add(v)       # remember what the limit hid
                return None
            for e in adj[u]:
                v = e >> 4
                if seen.get(v, 1 << 30) > d + 1:          # not seen, or seen deeper
                    seen[v] = d + 1; stats['gen'] += 1
                    r = dls(v, d + 1, limit)
                    if r is not None:
                        r.append(u); return r
            return None
        r = dls(s, 0, limit)
        tot['gen'] += stats['gen']; tot['exp'] += stats['exp']
        tot['maxstack'] = max(tot.get('maxstack', 0), stats['maxd'] + 1)
        tot['maxseen'] = max(tot.get('maxseen', 0), len(seen))
        tot['iters'] = limit + 1
        if r is not None:
            p = r[::-1]
            tot['maxf'] = tot['maxstack']
            return p, path_cost(adj, p), tot
        if not any(v not in seen for v in pruned):                                 # whole reachable graph exhausted
            tot['maxf'] = tot['maxstack']
            return None, None, tot

ALGOS = [("BFS", bfs), ("UCS", ucs), ("DFS", dfs), ("IDDFS", iddfs)]

def valid(adj, p, s, g):
    if p is None: return False
    if p[0] != s or p[-1] != g: return False
    return all(any(e >> 4 == b for e in adj[a]) for a, b in zip(p, p[1:]))

def pick_goal(adj, seed, n, island):
    """Goal = random node whose BFS depth is in the upper-middle range (a far ward)."""
    rng = random.Random(seed * 7 + 1)
    return rng.randrange(1, n - island)

def timed(fn, adj, s, g, runs):
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); out = fn(adj, s, g); ts.append(time.perf_counter() - t)
    return out, sum(ts) / len(ts)
