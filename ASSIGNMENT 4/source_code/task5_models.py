# Task 5: Categorical Naive Bayes from scratch; sklearn SVM comparison template
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,log_loss
from sklearn.naive_bayes import GaussianNB
from time import perf_counter

def fit_nb(X,y,alpha=1.0):
    classes=np.unique(y); prior={c:np.mean(y==c) for c in classes}
    cond={}
    for j in range(X.shape[1]):
        values=np.unique(X[:,j])
        for c in classes:
            yc=X[y==c,j]
            for v in values:
                cond[(j,c,v)]=(np.sum(yc==v)+alpha)/(len(yc)+alpha*len(values))
    return classes,prior,cond
def predict_nb(model,X):
    classes,prior,cond=model; scores=[]
    for row in X:
        scores.append([np.log(prior[c])+sum(np.log(max(cond.get((j,c,v),1e-12),1e-12))
                       for j,v in enumerate(row)) for c in classes])
    s=np.array(scores); s-=s.max(axis=1,keepdims=True); p=np.exp(s); return p/p.sum(axis=1,keepdims=True)
data=load_breast_cancer(); X=(data.data>np.median(data.data,axis=0)).astype(int); y=data.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=31)
for a in [0,.1,1,10]:
    m=fit_nb(Xtr,ytr,a); probs=predict_nb(m,Xte); pred=m[0][np.argmax(probs,axis=1)]
    print("alpha",a,"accuracy/precision/recall/F1/logloss",
          round(accuracy_score(yte,pred),4),round(precision_score(yte,pred),4),
          round(recall_score(yte,pred),4),round(f1_score(yte,pred),4),round(log_loss(yte,probs),4))
for kernel,grid in [("linear",{"C":[.1,1,10]}),("rbf",{"C":[.1,1,10],"gamma":["scale",.01,.1]})]:
    start=perf_counter()
    gs=GridSearchCV(SVC(kernel=kernel),grid,cv=StratifiedKFold(5,shuffle=True,random_state=31),scoring="f1")
    gs.fit(Xtr,ytr); pred=gs.predict(Xte)
    print(kernel,"best",gs.best_params_,"SV",gs.best_estimator_.n_support_.sum(),
          "accuracy",round(accuracy_score(yte,pred),4),"F1",round(f1_score(yte,pred),4),
          "seconds",round(perf_counter()-start,3))
