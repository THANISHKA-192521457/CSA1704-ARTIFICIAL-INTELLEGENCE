"""
Assignment 3 CO3 – compact reference implementations.
Educational starter code: extend and test each routine against the full report.
The assignment requires separate experiments and raw outputs; this file is not
a substitute for those measurements.
"""
from itertools import product

def truth_table_entails(symbols, kb, query):
    """kb and query are callables mapping {symbol: bool} -> bool."""
    models = 0
    countermodels = 0
    examined = 0
    for values in product([False, True], repeat=len(symbols)):
        model = dict(zip(symbols, values))
        examined += 1
        if kb(model):
            models += 1
            if not query(model):
                countermodels += 1
    return {
        "entailed": models > 0 and countermodels == 0,
        "kb_models": models,
        "countermodels": countermodels,
        "assignments_examined": examined,
    }

def modus_ponens(p, implication):
    """Return consequent if implication is represented as (antecedent, consequent)."""
    antecedent, consequent = implication
    return consequent if p == antecedent else None

def and_elimination(pair):
    return tuple(pair)

def unit_resolution(clause, unit):
    """Simple propositional clauses are sets of signed strings, e.g. {'P','Q'}."""
    if unit.startswith("~"):
        positive = unit[1:]
        if positive in clause:
            return set(clause) - {positive}
    else:
        negative = "~" + unit
        if negative in clause:
            return set(clause) - {negative}
    return None

def occurs(var, term):
    if var == term:
        return True
    if isinstance(term, tuple):
        return any(occurs(var, arg) for arg in term[1:])
    return False

def walk(term, subst):
    while isinstance(term, str) and term in subst:
        term = subst[term]
    return term

def unify(x, y, subst=None):
    """Terms: variable strings begin with '?'; compound term is (functor, arg1,...)."""
    if subst is None:
        subst = {}
    x, y = walk(x, subst), walk(y, subst)
    if x == y:
        return subst
    if isinstance(x, str) and x.startswith("?"):
        if occurs(x, y):
            return None
        result = dict(subst); result[x] = y
        return result
    if isinstance(y, str) and y.startswith("?"):
        if occurs(y, x):
            return None
        result = dict(subst); result[y] = x
        return result
    if isinstance(x, tuple) and isinstance(y, tuple) and len(x) == len(y) and x[0] == y[0]:
        result = dict(subst)
        for a, b in zip(x[1:], y[1:]):
            result = unify(a, b, result)
            if result is None:
                return None
        return result
    return None

def apply_substitution(term, subst):
    term = walk(term, subst)
    if isinstance(term, tuple):
        return (term[0],) + tuple(apply_substitution(x, subst) for x in term[1:])
    return term

def forward_chain(initial_facts, rules):
    """
    rules: list of (body_list, head); facts/goals are hashable tuples.
    This basic version scans rules repeatedly; add predicate indexing for scale.
    """
    known = set(initial_facts)
    firings = 0
    changed = True
    while changed:
        changed = False
        for body, head in rules:
            if all(item in known for item in body) and head not in known:
                known.add(head); firings += 1; changed = True
    return known, firings

def backward_chain(goal, facts, rules, depth=0, max_depth=100, visiting=None):
    """Ground propositional-style demonstration with loop/depth protection."""
    if visiting is None:
        visiting = set()
    if goal in facts:
        return True
    if depth >= max_depth or goal in visiting:
        return False
    visiting = set(visiting); visiting.add(goal)
    for body, head in rules:
        if head == goal and all(backward_chain(g, facts, rules, depth+1,
                                               max_depth, visiting) for g in body):
            return True
    return False

if __name__ == "__main__":
    # Small sanity tests
    result = unify(("Knows", "?x", "John"), ("Knows", "Mary", "?y"))
    print("Unifier:", result)
    print("Occurs-check failure:", unify("?x", ("f", "?x")))
    syms = ["P", "Q"]
    kb = lambda m: m["P"] and (not m["P"] or m["Q"])
    q = lambda m: m["Q"]
    print("Model-check result:", truth_table_entails(syms, kb, q))
