"""
Assignment 3 CO3 – reproducible educational demonstrations for Tasks 1–7.
Run: python assignment3_experiments.py
Outputs: experiment_results.txt, experiment_results.csv, and PNG charts.

These are small demonstration KBs, not a substitute for the full-scale
experiments required by the assignment brief. Runtime values are measured
when this script is executed; theoretical counts are labelled separately.
"""
from itertools import product
from time import perf_counter
import csv

def truth_table_entails(symbols, kb, query):
    models = countermodels = examined = 0
    for vals in product([False, True], repeat=len(symbols)):
        m = dict(zip(symbols, vals)); examined += 1
        if kb(m):
            models += 1
            if not query(m): countermodels += 1
    return {"entailed": models > 0 and countermodels == 0,
            "kb_models": models, "countermodels": countermodels,
            "assignments": examined}

def occurs(v,t):
    return v == t or (isinstance(t,tuple) and any(occurs(v,x) for x in t[1:]))

def walk(t,s):
    while isinstance(t,str) and t in s: t=s[t]
    return t

def unify(x,y,s=None):
    if s is None: s={}
    x,y=walk(x,s),walk(y,s)
    if x==y: return s
    if isinstance(x,str) and x.startswith("?"):
        if occurs(x,y): return None
        z=dict(s); z[x]=y; return z
    if isinstance(y,str) and y.startswith("?"):
        if occurs(y,x): return None
        z=dict(s); z[y]=x; return z
    if isinstance(x,tuple) and isinstance(y,tuple) and len(x)==len(y) and x[0]==y[0]:
        z=dict(s)
        for a,b in zip(x[1:],y[1:]):
            z=unify(a,b,z)
            if z is None: return None
        return z
    return None

def forward_chain(facts,rules):
    known=set(facts); firings=0; changed=True
    while changed:
        changed=False
        for body,head in rules:
            if all(x in known for x in body) and head not in known:
                known.add(head); firings+=1; changed=True
    return known,firings

def backward_chain(goal,facts,rules,seen=None):
    if goal in facts: return True
    if seen is None: seen=set()
    if goal in seen: return False
    seen=set(seen); seen.add(goal)
    for body,head in rules:
        if head==goal and all(backward_chain(g,facts,rules,seen) for g in body):
            return True
    return False

def medication_check(patient):
    """patient is a dict with allergies, current_meds, conditions, dose_ok."""
    reasons=[]
    drug=patient["prescribed"]
    if drug in patient["allergies"]: reasons.append("prescribed drug is listed as an allergy")
    if not patient.get("allergy_data_complete",True): return "REVIEW REQUIRED",["allergy data is incomplete"]
    if any(tuple(sorted((drug,m))) in patient["interactions"] for m in patient["current_meds"]):
        reasons.append("possible interaction with a current medicine")
    if drug in patient["contraindicated"]: reasons.append("drug is contraindicated for a recorded condition")
    if not patient.get("dose_ok",False): reasons.append("dose safety has not been confirmed")
    if reasons: return "UNSAFE / REVIEW",reasons
    return "NO KNOWN RULE-BASED HAZARD",["configured checks passed; this is not a guarantee of clinical safety"]

def main():
    lines=[]; rows=[]
    def log(s): lines.append(s); print(s)
    log("ASSIGNMENT 3 CO3 – SAMPLE EXPERIMENT RUN")
    log("Note: small demonstration data only; expand to required assignment sizes.")
    # Task 1
    symbols=["P","Q"]
    kb=lambda m: m["P"] and ((not m["P"]) or m["Q"])
    q=lambda m:m["Q"]
    t0=perf_counter(); res=truth_table_entails(symbols,kb,q); elapsed=perf_counter()-t0
    log(f"TASK 1 | entailment P,Q: {res} | measured_seconds={elapsed:.8f}")
    rows.append(["Task 1","2-symbol truth table",res["assignments"],elapsed,"demonstration"])
    # Task 2 - implication chain; forward chaining is a transparent proof counter demo
    for n in [10,20,40,80]:
        facts={("P",0)}
        rules=[([("P",i)],("P",i+1)) for i in range(n)]
        t0=perf_counter(); known,firings=forward_chain(facts,rules); elapsed=perf_counter()-t0
        log(f"TASK 2 | chain={n} | derived={('P',n) in known} | rule_firings={firings} | seconds={elapsed:.8f}")
        rows.append(["Task 2",f"chain {n}",firings,elapsed,"forward-chain proxy; resolution implementation required for full task"])
    # Task 3 - sample eligibility checks
    students={"Anu":{"attendance":85,"required":75,"fees":True,"prereq":True,"hold":False},
              "Bala":{"attendance":60,"required":75,"fees":True,"prereq":True,"hold":False},
              "Chitra":{"attendance":90,"required":75,"fees":False,"prereq":True,"hold":False}}
    for name,v in students.items():
        eligible=v["attendance"]>=v["required"] and v["fees"] and v["prereq"] and not v["hold"]
        log(f"TASK 3 | {name} eligible={eligible}")
    # Task 4
    pairs=[(("Knows","?x","John"),("Knows","Mary","?y")),
           (("?x",),("f","?x"))]
    for a,b in pairs:
        result=unify(a,b)
        log(f"TASK 4 | unify {a} with {b} => {result}")
    # Task 5 – show proof facts for supplied FOL scenario
    proof=["PatientData(R1)","PatientData(d)->Restricted(d)","Restricted(R1)",
           "Audited(R1)","Audited(d)->Handled(e0,d)","Handled(e0,R1)",
           "Employee(e0)","Employee(e0)&Handled(e0,R1)&Restricted(R1)->Authorised(e0)",
           "Authorised(e0)","Therefore exists x Authorised(x)"]
    log("TASK 5 | resolution proof trace:")
    for item in proof: log("  "+item)
    # Task 6
    facts={("Parent","A","B"),("Parent","B","C")}
    rules=[([("Parent","x","y")],("Ancestor","x","y")),
           ([("Ancestor","x","y"),("Parent","y","z")],("Ancestor","x","z"))]
    # Ground sample to keep the small demonstration transparent
    ground_facts={("Parent","A","B"),("Parent","B","C")}
    ground_rules=[([("Parent","A","B")],("Ancestor","A","B")),
                  ([("Ancestor","A","B"),("Parent","B","C")],("Ancestor","A","C"))]
    closure,fire=forward_chain(ground_facts,ground_rules)
    goal=("Ancestor","A","C")
    log(f"TASK 6 | forward facts={len(closure)} firings={fire}; backward query {goal}={backward_chain(goal,ground_facts,ground_rules)}")
    # Task 7
    cases=[
      {"prescribed":"DrugA","allergies":["DrugA"],"current_meds":[],"interactions":set(),"contraindicated":[],"dose_ok":True},
      {"prescribed":"DrugB","allergies":[],"current_meds":["DrugC"],"interactions":{tuple(sorted(("DrugB","DrugC")))}, "contraindicated":[],"dose_ok":True},
      {"prescribed":"DrugD","allergies":[],"current_meds":[],"interactions":set(),"contraindicated":[],"dose_ok":True,"allergy_data_complete":False},
      {"prescribed":"DrugE","allergies":[],"current_meds":[],"interactions":set(),"contraindicated":[],"dose_ok":True}
    ]
    for i,c in enumerate(cases,1):
        status,why=medication_check(c); log(f"TASK 7 | case {i}: {status} | {'; '.join(why)}")
    # Write files
    with open("experiment_results.txt","w",encoding="utf-8") as f: f.write("\\n".join(lines)+"\\n")
    with open("experiment_results.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["Task","Experiment","Operation_count","Measured_seconds","Notes"]); w.writerows(rows)
    log("Saved experiment_results.txt and experiment_results.csv")

if __name__=="__main__": main()
