from drone_sim import run_scenarios, swarm_demo
from expert_system import diagnose
from edge_demo import run as edge_run
from chatbot import classify
print("=== DRONE ===")
for x in run_scenarios(): print(x)
print("=== SWARM ===")
for x in swarm_demo(): print(x)
print("=== EXPERT SYSTEM ===")
print(diagnose({"leak":"yes","pressure":"low"}))
print("=== EDGE DEMO ===")
edge_run()
print("=== CHATBOT ===")
for q in ["Which alerts are open?","Why was this flagged?","tell me a joke"]:
    print(q, classify(q))
