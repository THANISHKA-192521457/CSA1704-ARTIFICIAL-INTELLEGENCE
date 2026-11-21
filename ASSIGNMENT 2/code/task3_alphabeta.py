"""TASK 3 - How tree structure (move ordering) controls the work of alpha-beta pruning.
Trees are nested lists; leaves are integers. Root is a MAX node. Uniform branching b, depth d."""
import random, json, sys, statistics as st
sys.setrecursionlimit(10000)

def make_tree(b, d, rng):
    leaves = list(range(b**d)); rng.shuffle(leaves)      # distinct leaf values => no ties
    it = iter(leaves)
    def build(depth):
        if depth == 0: return next(it)
        return [build(depth-1) for _ in range(b)]
    return build(d)

def minimax_value(node, is_max):
    if isinstance(node, int): return node
    vals = [minimax_value(c, not is_max) for c in node]
    return max(vals) if is_max else min(vals)

def reorder(node, is_max, best_first):
    """Return (value, reordered subtree). best_first=True -> strongest move first (perfect ordering);
    False -> weakest move first (adversarial ordering)."""
    if isinstance(node, int): return node, node
    kids = [reorder(c, not is_max, best_first) for c in node]
    desc = (is_max == best_first)          # MAX & best-first -> descending ; MIN & best-first -> ascending
    kids.sort(key=lambda t: t[0], reverse=desc)
    vals = [k[0] for k in kids]
    return (max(vals) if is_max else min(vals)), [k[1] for k in kids]

class Counter:
    def __init__(self): self.nodes = 0; self.leaves = 0; self.cutoffs = 0

def minimax(node, is_max, c):
    c.nodes += 1
    if isinstance(node, int): c.leaves += 1; return node
    vals = [minimax(ch, not is_max, c) for ch in node]
    return max(vals) if is_max else min(vals)

def alphabeta(node, is_max, alpha, beta, c):
    c.nodes += 1
    if isinstance(node, int): c.leaves += 1; return node
    if is_max:
        v = -10**9
        for ch in node:
            v = max(v, alphabeta(ch, False, alpha, beta, c)); alpha = max(alpha, v)
            if beta <= alpha: c.cutoffs += 1; break
        return v
    v = 10**9
    for ch in node:
        v = min(v, alphabeta(ch, True, alpha, beta, c)); beta = min(beta, v)
        if beta <= alpha: c.cutoffs += 1; break
    return v

def predict_best(b, d):
    """Knuth-Moore minimal tree: PV node(N1) = 1 PV child + (b-1) cut children; cut node(N2) = 1 'all' child;
    'all' node(N3) = b cut children. Returns (leaves, total nodes)."""
    n1 = n2 = n3 = 1; t1 = t2 = t3 = 1
    for _ in range(d):
        n1, n2, n3 = n1 + (b-1)*n2, n3, b*n2
        t1, t2, t3 = 1 + t1 + (b-1)*t2, 1 + t3, 1 + b*t2
    return n1, t1

def engineered_no_cutoff(b, d):
    """Interval construction: every node owns a block of consecutive distinct integers.
    MAX node: child i gets block i (ascending -> each child improves alpha, worst first).
    MIN node: child i gets block b-1-i (descending -> each child improves beta, worst first).
    Blocks nest, so alpha < every value below a node < beta and beta<=alpha never becomes true."""
    def build(depth, is_max, start):
        if depth == 0: return start
        size = b**(depth-1)
        order = range(b) if is_max else range(b-1, -1, -1)
        return [build(depth-1, not is_max, start + i*size) for i in order]
    return build(d, True, 0)

def run(tree, b, d):
    cm, ca = Counter(), Counter()
    v1 = minimax(tree, True, cm); v2 = alphabeta(tree, True, -10**9, 10**9, ca)
    assert v1 == v2, "alpha-beta must return the minimax value"
    return v1, cm, ca

if __name__ == '__main__':
    b, d = 4, 8
    pl, pt = predict_best(b, d)
    PRED = {  # predictions fixed BEFORE any experiment is executed
      'best_ordered':  dict(leaves=pl, nodes=pt, cutoffs='many'),
      'worst_by_value_sort': dict(leaves=b**d, nodes=(b**(d+1)-1)//(b-1), cutoffs=0),
      'engineered_no_cutoff': dict(leaves=b**d, nodes=(b**(d+1)-1)//(b-1), cutoffs=0),
      'random_order':  dict(leaves='~b^(3d/4)=%d (range 3000-9000)' % round(b**(0.75*d)), nodes='~1.3 x leaves', cutoffs='many'),
      'equal_leaves':  dict(leaves='close to best case (%d)' % pl, nodes='close to %d' % pt, cutoffs='very many'),
    }
    print('PREDICTIONS', json.dumps(PRED, indent=1))
    rng = random.Random(42); base = make_tree(b, d, rng)
    rows = []
    def record(name, tree, bb=b, dd=d):
        v, cm, ca = run(tree, bb, dd)
        rows.append(dict(test=name, b=bb, d=dd, minimax_nodes=cm.nodes, minimax_leaves=cm.leaves,
                         ab_nodes=ca.nodes, ab_leaves=ca.leaves, ab_cutoffs=ca.cutoffs,
                         saving_pct=round(100*(1-ca.nodes/cm.nodes), 2), value=v))
        print(rows[-1])
    record('best_ordered', reorder(base, True, True)[1])
    record('worst_by_value_sort', reorder(base, True, False)[1])
    record('engineered_no_cutoff', engineered_no_cutoff(b, d))
    def equal_tree(depth):
        return 7 if depth == 0 else [equal_tree(depth-1) for _ in range(b)]
    record('equal_leaves', equal_tree(d))
    # random ordering: 30 independent random trees
    rs = []
    for s in range(30):
        t = make_tree(b, d, random.Random(100+s)); v, cm, ca = run(t, b, d); rs.append((ca.nodes, ca.leaves, ca.cutoffs, cm.nodes))
    rows.append(dict(test='random_order_mean_of_30', b=b, d=d, minimax_nodes=rs[0][3], minimax_leaves=b**d,
                     ab_nodes=round(st.mean(r[0] for r in rs)), ab_leaves=round(st.mean(r[1] for r in rs)),
                     ab_cutoffs=round(st.mean(r[2] for r in rs)),
                     saving_pct=round(100*(1-st.mean(r[0] for r in rs)/rs[0][3]), 2),
                     leaves_min=min(r[1] for r in rs), leaves_max=max(r[1] for r in rs), value=None))
    print(rows[-1])
    # scaling with depth: b=4, d=4,6,8 (shows that (b,d) alone does not determine the work)
    scal = []
    for dd in (4, 6, 8):
        tr = make_tree(4, dd, random.Random(5))
        pb = predict_best(4, dd)
        for name, tree in (('best', reorder(tr, True, True)[1]), ('worst_value_sort', reorder(tr, True, False)[1]), ('engineered', engineered_no_cutoff(4, dd)), ('random', tr)):
            v, cm, ca = run(tree, 4, dd)
            scal.append(dict(d=dd, ordering=name, minimax_leaves=cm.leaves, ab_leaves=ca.leaves, ab_nodes=ca.nodes, cutoffs=ca.cutoffs,
                             predicted_best_leaves=pb[0] if name == 'best' else None))
    for r in scal: print(r)
    json.dump(dict(predictions=PRED, rows=rows, scaling=scal), open('../results/task3_results.json', 'w'), indent=1)
