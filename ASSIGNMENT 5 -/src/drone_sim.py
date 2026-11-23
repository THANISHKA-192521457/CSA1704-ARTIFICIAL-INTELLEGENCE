"""Hybrid parcel-drone and decentralised swarm grid demonstration."""
from dataclasses import dataclass
from collections import deque
import random

W, H = 8, 6
START, GOAL = (0, 0), (7, 5)
NO_FLY = {(3, 1), (3, 2), (3, 3), (5, 4)}
OBSTACLES = {(2, 0), (4, 2)}

@dataclass
class Drone:
    pos: tuple = START
    battery: int = 24
    status: str = "ACTIVE"
    memory: tuple = START

def neighbours(p):
    x, y = p
    return [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]

def safe(p):
    return 0 <= p[0] < W and 0 <= p[1] < H and p not in NO_FLY and p not in OBSTACLES

def bfs(start, goal):
    q = deque([(start, [start])]); seen = {start}
    while q:
        p, path = q.popleft()
        if p == goal: return path
        for n in neighbours(p):
            if safe(n) and n not in seen:
                seen.add(n); q.append((n, path+[n]))
    return None

def hybrid_step(agent, step, sudden_obstacle=None):
    """Safety layer overrides route planner; then battery/goal policy."""
    percept = {"position": agent.pos, "battery": agent.battery,
               "obstacle": sudden_obstacle, "goal": GOAL}
    if sudden_obstacle is not None and sudden_obstacle in neighbours(agent.pos):
        layer, action = "REACTIVE_SAFETY", "HOVER/REPLAN"
        OBSTACLES.add(sudden_obstacle)
    elif agent.battery <= 4:
        layer, action = "DELIBERATIVE", "RETURN/LAND"
        agent.status = "RETURNING"
    else:
        path = bfs(agent.pos, GOAL)
        if path and len(path) > 1:
            layer, action = "PLANNER", f"MOVE {path[1]}"
            agent.pos = path[1]
        else:
            layer, action = "SAFETY", "LAND"
            agent.status = "LANDED"
    agent.memory = agent.pos
    agent.battery = max(0, agent.battery-1)
    return {"step":step,"percept":percept,"state":agent.memory,
            "layer":layer,"action":action,"battery_after":agent.battery}

def run_scenarios():
    # Reset mutable obstacle set to deterministic baseline.
    OBSTACLES.clear(); OBSTACLES.update({(2,0),(4,2)})
    a = Drone()
    logs=[]
    for t in range(1,9):
        sudden = (1,0) if t == 3 else None
        logs.append(hybrid_step(a,t,sudden))
        if a.status == "LANDED": break
    return logs

def swarm_demo():
    """Local frontier rule: choose nearest unvisited safe cell; avoid neighbour positions."""
    grid = [(x,y) for y in range(5) for x in range(5)]
    drones = [(0,0),(4,0),(2,4)]
    visited=set(drones); trace=[]
    for tick in range(1,7):
        next_positions=[]
        for i,p in enumerate(drones):
            occupied=set(drones)-{p}
            candidates=[n for n in neighbours(p) if 0<=n[0]<5 and 0<=n[1]<5 and n not in visited and n not in occupied]
            if candidates:
                # deterministic frontier preference: smallest Manhattan distance to centre, then x/y
                target=min(candidates,key=lambda z:(abs(z[0]-2)+abs(z[1]-2),z[1],z[0]))
                next_positions.append(target); visited.add(target)
            else: next_positions.append(p)
        drones=next_positions
        trace.append((tick,tuple(drones),len(visited)))
    return trace

if __name__ == "__main__":
    for row in run_scenarios(): print(row)
    print("SWARM")
    for row in swarm_demo(): print(row)
