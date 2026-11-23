"""Deterministic synthetic edge-vs-full model benchmark (demonstration only)."""
import random, time, statistics, json
def sigmoid(z): return 1/(1+pow(2.718281828,-max(-30,min(30,z))))
def score(x, weights):
    return sum(a*b for a,b in zip(x,weights))
def make_data(n=1200, seed=17):
    r=random.Random(seed); rows=[]
    for _ in range(n):
        x=[r.random() for _ in range(8)]
        y=int((x[0]*0.9+x[1]*0.7+x[3]*0.5+x[5]*0.3)>1.15)
        rows.append((x,y))
    return rows
def metrics(y_true,y_pred):
    tp=sum(a==1 and b==1 for a,b in zip(y_true,y_pred))
    tn=sum(a==0 and b==0 for a,b in zip(y_true,y_pred))
    fp=sum(a==0 and b==1 for a,b in zip(y_true,y_pred))
    fn=sum(a==1 and b==0 for a,b in zip(y_true,y_pred))
    p=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
    f1=2*p*rec/(p+rec) if p+rec else 0
    return {"accuracy":(tp+tn)/len(y_true),"precision":p,"recall":rec,"f1":f1,
            "confusion_matrix":[[tn,fp],[fn,tp]]}
def run():
    data=make_data()
    train=data[:800]; test=data[800:] # deterministic split; synthetic demonstration
    # Transparent fixed coefficients stand in for a pre-trained model.
    full_w=[1.8,1.4,0.1,1.0,0.1,0.6,0.1,0.1]
    edge_w=[1.5,1.1,0.0,0.8,0.0,0.4,0.0,0.0] # feature reduction / quantised-like compact model
    yt=[y for _,y in test]
    full=[int(score(x,full_w)>1.65) for x,y in test]
    edge=[int(score(x,edge_w)>1.45) for x,y in test]
    out={"dataset":"synthetic intrusion-like demo; not a public benchmark",
         "train_n":len(train),"test_n":len(test),
         "full":metrics(yt,full),"edge":metrics(yt,edge),
         "demo_classifier_coefficients_full":len(full_w),"demo_classifier_coefficients_edge":len(edge_w),
         "illustrative_dense_full_model_parameters":9409,
         "illustrative_compact_edge_model_parameters":161,
         "illustrative_full_fp_ops_per_sample":9216,
         "illustrative_edge_fp_ops_per_sample":288,
         "illustrative_full_weight_memory_float32_bytes":37636,
         "illustrative_edge_weight_memory_int8_bytes":161}
    print(json.dumps(out,indent=2))
    return out
if __name__=="__main__": run()
