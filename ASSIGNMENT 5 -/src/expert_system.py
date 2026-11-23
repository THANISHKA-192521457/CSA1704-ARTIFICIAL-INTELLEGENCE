"""Generic rule-shell demonstration with HOW/WHY explanations."""
RULES = [
 {"id":"P01","if":{"vibration":"high","temperature":"high"},"then":"bearing_failure","priority":5},
 {"id":"P02","if":{"pressure":"low","flow":"low"},"then":"cavitation","priority":4},
 {"id":"P03","if":{"motor_current":"high","speed":"low"},"then":"motor_overload","priority":5},
 {"id":"P04","if":{"leak":"yes","pressure":"low"},"then":"seal_leak","priority":6},
 {"id":"P05","if":{"noise":"high","vibration":"high"},"then":"bearing_failure","priority":3},
 {"id":"P06","if":{"temperature":"high","pressure":"normal"},"then":"cooling_fault","priority":2},
 {"id":"P07","if":{"flow":"low","valve":"closed"},"then":"valve_blockage","priority":4},
 {"id":"P08","if":{"motor_current":"normal","speed":"low"},"then":"mechanical_drag","priority":2},
 {"id":"P09","if":{"leak":"yes","flow":"low"},"then":"seal_leak","priority":5},
 {"id":"P10","if":{"pressure":"low","vibration":"high"},"then":"cavitation","priority":3},
 {"id":"P11","if":{"temperature":"high","motor_current":"high"},"then":"motor_overload","priority":4},
 {"id":"P12","if":{"noise":"high","pressure":"normal"},"then":"bearing_wear","priority":2},
 {"id":"P13","if":{"flow":"normal","pressure":"low"},"then":"sensor_fault","priority":1},
 {"id":"P14","if":{"valve":"closed","pressure":"high"},"then":"valve_blockage","priority":4},
 {"id":"P15","if":{"temperature":"normal","vibration":"normal"},"then":"healthy","priority":1},
]
def diagnose(facts, strategy="specificity"):
    fired=[r for r in RULES if all(facts.get(k)==v for k,v in r["if"].items())]
    if not fired: return {"diagnosis":"insufficient_evidence","fired":[],"how":"No rule matched.","why":"The supplied facts do not satisfy any rule."}
    if strategy=="priority":
        chosen=max(fired,key=lambda r:r["priority"])
    elif strategy=="recency":
        chosen=max(fired,key=lambda r:max([list(facts).index(k) for k in r["if"] if k in facts] or [-1]))
    else:
        chosen=max(fired,key=lambda r:(len(r["if"]),r["priority"]))
    return {"diagnosis":chosen["then"],"fired":[r["id"] for r in fired],
            "how":f"{chosen['id']} matched {chosen['if']} and concluded {chosen['then']}.",
            "why":f"Observed facts supporting the selected rule: {', '.join(f'{k}={v}' for k,v in chosen['if'].items())}."}
def add_rule(rule):
    """Knowledge acquisition: update rule base without changing inference engine."""
    RULES.append(rule)
if __name__=="__main__":
    cases=[
      {"vibration":"high","temperature":"high"},
      {"pressure":"low","flow":"low"},
      {"leak":"yes","pressure":"low"},
      {"noise":"high","vibration":"high"},
      {"temperature":"normal","vibration":"normal"},
    ]
    for c in cases: print(c, "=>", diagnose(c))
