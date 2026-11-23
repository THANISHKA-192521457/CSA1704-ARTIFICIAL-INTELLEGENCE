# Task 1: Demonstrate training accuracy optimism with a fixed seed
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
X, y = make_classification(n_samples=500, n_features=20, n_informative=4,
                           n_redundant=4, flip_y=0.12, random_state=17)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=17)
model = DecisionTreeClassifier(random_state=17)
model.fit(Xtr, ytr)
train_acc = accuracy_score(ytr, model.predict(Xtr))
cv = cross_val_score(model, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=17))
pred = model.predict(Xte)
print("Training accuracy:", round(train_acc, 4))
print("5-fold CV accuracy:", round(cv.mean(), 4))
print("Held-out accuracy:", round(accuracy_score(yte, pred), 4))
print("Optimistic gap (train - test):", round(train_acc-accuracy_score(yte,pred), 4))
print("Precision/Recall/F1:", *(round(v,4) for v in (
 precision_score(yte,pred,zero_division=0), recall_score(yte,pred,zero_division=0),
 f1_score(yte,pred,zero_division=0))))
print("Confusion matrix:\\n", confusion_matrix(yte,pred))
