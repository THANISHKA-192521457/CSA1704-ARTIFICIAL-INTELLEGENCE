# Task 4: NumPy backpropagation, finite-difference check and XOR
import numpy as np
sig=lambda z: 1/(1+np.exp(-z))
def forward(x,W1,b1,W2,b2):
    z1=W1@x+b1; a1=sig(z1); z2=W2@a1+b2; a2=sig(z2)
    return z1,a1,z2,a2
def gradients(x,y,W1,b1,W2,b2):
    z1,a1,z2,a2=forward(x,W1,b1,W2,b2)
    d2=(a2-y)*a2*(1-a2); d1=(W2.T@d2)*a1*(1-a1)
    return (d1@x.T,d1,d2@a1.T,d2),a2
# Task 4.4 trace (2 hidden units, one output)
x=np.array([[1.],[0.]])
W1=np.array([[.5,-.4],[.3,.8]]); b1=np.zeros((2,1))
W2=np.array([[.6,-.2]]); b2=np.zeros((1,1))
(z1,a1,z2,a2)=forward(x,W1.T,b1,W2,b2)
print("hidden activations:",a1.ravel(),"output:",a2.ravel())
g,_=gradients(x,1.,W1.T,b1,W2,b2)
print("analytic gradient norms:",[float(np.linalg.norm(v)) for v in g])
# XOR train a 2-8-1 sigmoid network
rng=np.random.default_rng(8); X=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
Y=np.array([[0.],[1.],[1.],[0.]])
A=rng.normal(0,.7,(8,2)); c=np.zeros((8,1)); B=rng.normal(0,.7,(1,8)); d=np.zeros((1,1))
for epoch in range(15000):
    z1=X@A.T+c.T; h=sig(z1); out=sig(h@B.T+d.T)
    delta2=(out-Y)*out*(1-out); delta1=(delta2@B)*h*(1-h)
    B-=.8*(delta2.T@h)/4; d-=.8*delta2.mean(axis=0).reshape(1,1)
    A-=.8*(delta1.T@X)/4; c-=.8*delta1.mean(axis=0).reshape(8,1)
print("XOR predictions:",np.round(sig(sig(X@A.T+c.T)@B.T+d.T).ravel(),3))
