import sys
sys.path.append('../')
import chorales as ch
import numpy as np
import LSTM as l
import block as blk
import matplotlib.pyplot as plt


# X = np.array([[0,0,1,0,1,0,1,0,1,0,1,0,1],[0,1,0,1,0,1,0,1,0,1,0,1,0]])
# Y = np.array([[0,1,0,1,0,1,0,1,0,1,0,1,0],[0,0,1,0,1,0,1,0,1,0,1,0,1]])

#X = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,1,1,1,1,1,1,1,1,1,1,1],[0,1,1,1,1,1,1,1,1,1,1,1,1]])
# X = np.zeros((4,1000))
# for i in range(1000):
# 	if i > 0:
# 		X[1,i] = 0;
# 		X[2,i] = 1;
# 		X[3,i] = 1;
#Y = X


def gen_check_data():
    check = []
    temp = np.zeros((5,13))
    for i in range(len(temp[0])):
        k = i % 10
        if k > 4: k = (-k)%5-1
        temp[k][i] = 1
    
    check.append(temp)
    return check

x = gen_check_data()

X = np.zeros((5,13))
Y = np.zeros((5,13))
X[:,1:] = x[0][:,:12]+.001
Y[:,1:] = x[0][:,1:]+.001


N = 5
M = 5
T = 12
itr = 100
learn_rate = .05

 
# d = l.deltas(N,M,T)

b = blk.block(N,M,T)

L = np.zeros(itr)
Yhat = []
Out = np.zeros((N,T,itr))

# for i in range(10):
# 	Yhat.append(l.forward_pass(X,b,N,M,T))
# 	d, L[i] = l.get_delta(X,Y,Yhat[i],b,d,N,M,T)
# 	b = l.update_weights(b,d,.05,N,M,T)

for i in range(itr):
	Yhat.append(l.forward_pass(X,b,N,M,T))
	L[i], b, Out[:,:,i] = l.backprop(X,Y,Yhat[i],b,N,M,T,learn_rate)
	print(i)
	print('------------------------')

plt.plot(L)
plt.show()
