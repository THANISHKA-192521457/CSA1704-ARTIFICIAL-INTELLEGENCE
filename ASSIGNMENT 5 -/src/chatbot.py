"""TF-IDF-like retrieval chatbot with cosine similarity and confidence fallback."""
import math, re
INTENTS={
 "open_alerts":["show open alerts","list active alerts","which alerts are open"],
 "why_flagged":["why was this flagged","explain this detection","reason for alert"],
 "system_status":["system status","is the detector running","show system health"],
 "drone_location":["where is the drone","show drone position","drone location"],
 "battery_status":["battery level","how much battery remains","drone battery"],
 "sensor_reading":["show sensor reading","air quality value","latest sensor data"],
 "help":["help","what can you do","available commands"],
 "exit":["exit","quit","close assistant"]
}
def tokens(s): return re.findall(r"[a-z]+",s.lower())
def vector(s,vocab):
    t=tokens(s); return [t.count(w) for w in vocab]
def cosine(a,b):
    den=math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    return sum(x*y for x,y in zip(a,b))/den if den else 0
TRAIN=[(phrase,intent) for intent,phrases in INTENTS.items() for phrase in phrases]
VOCAB=sorted({w for phrase,_ in TRAIN for w in tokens(phrase)})
def classify(text,threshold=0.22):
    q=vector(text,VOCAB)
    ranked=sorted(((cosine(q,vector(p,VOCAB)),intent,p) for p,intent in TRAIN),reverse=True)
    score,intent,phrase=ranked[0]
    if score<threshold: return {"intent":"out_of_scope","confidence":round(score,3),"response":"Sorry, I could not confidently map that request. Please rephrase or ask for help."}
    responses={
      "open_alerts":"There are 2 demonstration alerts open.",
      "why_flagged":"The sample event crossed the configured anomaly threshold.",
      "system_status":"The demonstration detector is running normally.",
      "drone_location":"The sample drone is at grid coordinate (2, 3).",
      "battery_status":"The sample drone battery is 68 percent.",
      "sensor_reading":"The latest demonstration sensor value is 42 units.",
      "help":"I can answer alerts, explanation, status, drone, battery and sensor questions.",
      "exit":"Assistant session ended."
    }
    return {"intent":intent,"confidence":round(score,3),"response":responses[intent]}
if __name__=="__main__":
    for q in ["Which alerts are open?","Why was this flagged?","tell me a joke"]:
        print(q,"=>",classify(q))
