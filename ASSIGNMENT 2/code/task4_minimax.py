"""TASK 4 - Depth-limited minimax with a hand-made evaluation function (Connect-4 family, b = #columns).
Theory: positions examined to depth d  N(d) = sum_{i=0..d} b^i = (b^(d+1)-1)/(b-1);
        evaluation calls (leaves)      L(d) = b^d   (fewer if the game ends earlier)."""
import random, time, json

class Game:
    def __init__(self, R, C, K=4):
        self.R, self.C, self.K = R, C, K
        self.g = [[0]*C for _ in range(R)]; self.h = [0]*C
        self.windows = []
        for r in range(R):
            for c in range(C):
                for dr, dc in ((0,1),(1,0),(1,1),(1,-1)):
                    er, ec = r+dr*(K-1), c+dc*(K-1)
                    if 0 <= er < R and 0 <= ec < C:
                        self.windows.append([(r+dr*i, c+dc*i) for i in range(K)])
    def moves(self):
        order = sorted(range(self.C), key=lambda c: abs(c-(self.C-1)/2))   # centre first (does not change plain-minimax work)
        return [c for c in order if self.h[c] < self.R]
    def play(self, c, p): self.g[self.h[c]][c] = p; self.h[c] += 1
    def undo(self, c): self.h[c] -= 1; self.g[self.h[c]][c] = 0
    def wins(self, c, p):               # did the piece just dropped in column c (by p) complete K in a row?
        r = self.h[c]-1; g = self.g
        for dr, dc in ((0,1),(1,0),(1,1),(1,-1)):
            n = 1
            for s in (1, -1):
                rr, cc = r+dr*s, c+dc*s
                while 0 <= rr < self.R and 0 <= cc < self.C and g[rr][cc] == p:
                    n += 1; rr += dr*s; cc += dc*s
            if n >= self.K: return True
        return False
    def full(self): return all(x == self.R for x in self.h)
    def evaluate(self, me=1):
        """Own evaluation: sum over every K-window that is not blocked: +/- {1,10,100} for 1,2,3 own pieces
        (K=4), plus a small centre-column bonus. Positive = good for player 1."""
        g = self.g; score = 0
        for w in self.windows:
            a = b = 0
            for r, c in w:
                v = g[r][c]
                if v == 1: a += 1
                elif v == 2: b += 1
            if a and not b: score += 10**(a-1)
            elif b and not a: score -= 10**(b-1)
        mid = self.C//2
        for r in range(self.R):
            v = g[r][mid]
            score += 3 if v == 1 else (-3 if v == 2 else 0)
        return score

WIN = 100000
class Stats:
    def __init__(self): self.nodes = 0; self.evals = 0; self.terminal = 0

def minimax(G, depth, player, last, st):
    """returns value from player 1's point of view; player in {1,2} is the side to move"""
    st.nodes += 1
    if last is not None and G.wins(last, 3-player):
        st.terminal += 1
        return (-WIN - depth) if player == 1 else (WIN + depth)     # earlier wins score higher
    ms = G.moves()
    if not ms: st.terminal += 1; return 0
    if depth == 0: st.evals += 1; return G.evaluate()
    best = -10**9 if player == 1 else 10**9
    for c in ms:
        G.play(c, player)
        v = minimax(G, depth-1, 3-player, c, st)
        G.undo(c)
        best = max(best, v) if player == 1 else min(best, v)
    return best

def choose(G, depth, player, rng, st):
    best, bm = None, []
    for c in G.moves():
        G.play(c, player)
        if G.wins(c, player): v = (WIN if player == 1 else -WIN)
        else: v = minimax(G, depth-1, 3-player, c, st)
        G.undo(c)
        better = best is None or (v > best if player == 1 else v < best)
        if better: best, bm = v, [c]
        elif v == best: bm.append(c)
    return rng.choice(bm)

def play_game(R, C, depth, minimax_first, seed, st):
    rng = random.Random(seed); G = Game(R, C); p = 1; me = 1 if minimax_first else 2
    while True:
        ms = G.moves()
        if not ms: return 0
        c = choose(G, depth, p, rng, st) if p == me else rng.choice(ms)
        G.play(c, p)
        if G.wins(c, p): return 1 if p == me else -1
        p = 3-p

if __name__ == '__main__':
    out = {}
    # ---- (a) node-count growth from the empty board
    growth = []
    for (R, C) in ((4, 5), (5, 6), (6, 7)):
        b = C
        for d in range(1, 8 if C == 5 else (7 if C == 6 else 6) + 1):
            G = Game(R, C); st = Stats(); t = time.perf_counter()
            minimax(G, d, 1, None, st); dt = time.perf_counter()-t
            growth.append(dict(board=f"{R}x{C}", b=b, depth=d, nodes=st.nodes, theory_nodes=(b**(d+1)-1)//(b-1),
                               eval_calls=st.evals, theory_leaves=b**d, terminal=st.terminal, time_s=round(dt, 3)))
            print(growth[-1], flush=True)
            if dt > 40: break
    out['growth'] = growth
    # ---- (b) playing strength: 20 games vs random player
    strength = []
    for (R, C, depths) in ((4, 5, (1, 2, 3, 4, 5)), (5, 6, (1, 2, 3, 4)), (6, 7, (1, 2, 3, 4))):
        for d in depths:
            st = Stats(); w = dr = l = 0; t = time.perf_counter()
            for gm in range(20):
                res = play_game(R, C, d, gm % 2 == 0, 1000+gm, st)
                w += res == 1; dr += res == 0; l += res == -1
            strength.append(dict(board=f"{R}x{C}", b=C, depth=d, wins=w, draws=dr, losses=l, win_rate_pct=5*w,
                                 nodes_total=st.nodes, eval_calls_total=st.evals, time_s=round(time.perf_counter()-t, 2)))
            print(strength[-1], flush=True)
    out['strength'] = strength
    json.dump(out, open('../results/task4_results.json', 'w'), indent=1)
