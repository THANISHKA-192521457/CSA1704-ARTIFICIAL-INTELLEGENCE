# Task 2: From-scratch categorical decision tree and sklearn verification
import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def entropy(y):
    counts=np.bincount(y)
    p=counts[counts>0]/len(y)
    return -np.sum(p*np.log2(p))

def best_split(X,y,criterion="entropy"):
    base=entropy(y) if criterion=="entropy" else 1-np.sum((np.bincount(y)/len(y))**2)
    best_gain=-1; best=None
    for j in range(X.shape[1]):
        weighted=0
        for v in np.unique(X[:,j]):
            idx=X[:,j]==v; yy=y[idx]
            if criterion=="entropy": score=entropy(yy)
            else:
                p=np.bincount(yy,minlength=2)/len(yy); score=1-np.sum(p*p)
            weighted += len(yy)/len(y)*score
        gain=base-weighted
        if gain>best_gain: best_gain,best=gain,j
    return best,best_gain

def build(X,y,depth=0,max_depth=3,min_samples=5,criterion="entropy"):
    majority=Counter(y).most_common(1)[0][0]
    if len(set(y))==1 or depth>=max_depth or len(y)<min_samples or X.shape[1]==0:
        return ("leaf",majority)
    j,g=best_split(X,y,criterion)
    if j is None or g<=1e-12: return ("leaf",majority)
    branches={}
    for v in np.unique(X[:,j]):
        m=X[:,j]==v
        branches[v]=build(X[m],y[m],depth+1,max_depth,min_samples,criterion)
    return ("node",j,majority,branches)

def predict_one(tree,row):
    if tree[0]=="leaf": return tree[1]
    _,j,default,branches=tree
    v=row[j]
    return predict_one(branches[v],row) if v in branches else default

def predict(tree,X): return np.array([predict_one(tree,r) for r in X])
data=load_breast_cancer()
# Discretise numeric features into binary categories using training-independent median for demonstration.
X=np.column_stack([data.data[:,j] > np.median(data.data[:,j]) for j in range(data.data.shape[1])]).astype(int)
y=data.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=21)
for crit in ["entropy","gini"]:
    tree=build(Xtr,ytr,max_depth=4,criterion=crit)
    p=predict(tree,Xte)
    print(crit, "accuracy/precision/recall/F1:",
          *(round(f(yte,p),4) for f in (accuracy_score,precision_score,recall_score,f1_score)))
    print("confusion matrix:\\n",confusion_matrix(yte,p))
sk=DecisionTreeClassifier(max_depth=4,criterion="entropy",random_state=21).fit(Xtr,ytr)
ps=sk.predict(Xte)
print("sklearn accuracy:",round(accuracy_score(yte,ps),4))
print("sklearn confusion matrix:\\n",confusion_matrix(yte,ps))
