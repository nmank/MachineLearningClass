#this is a one layer LSTM

import numpy as np;

def sig(x):
	y = 1./(1+np.exp(-x))
	return y

def dsig(x):
	# y = np.zeros(x.shape)
	# for i in range(x.size):
	# 	if x[i] < -350 :
	# 		y[i] = 0
	# 	else:
	# 		y[i] = (np.exp(-x[i]))/((1+np.exp(-x[i]))*(1+np.exp(-x[i])))
	y = sig(x)*(1-sig(x))
	return y

def dtanh(x):
	y = np.zeros(x.shape)
	for i in range(x.size):
		if x[i] < -300 or x[i] > 300:
			y[i] = 0
		else:
			y[i] = 1/(np.cosh(x[i])*np.cosh(x[i]))
	# y = 1. - x**2
	return y

def softmax(x):
	return np.divide(np.exp(x),np.sum(np.exp(x)))

#learning rate

#import data
#T = number of time frames
#N dimension of input data
#M dimension of target
#make sure the first column of x is blank



def forward_pass(X,blk,N,M,T):
	Yhat= np.zeros((M,T+1));

	blk.zb = np.zeros((M,T+2))
	blk.ib = np.zeros((M,T+2))
	blk.fb = np.zeros((M,T+2))
	blk.ob = np.zeros((M,T+2))
	blk.c = np.zeros((M,T+2))
	blk.z = np.zeros((M,T+2))
	blk.i = np.zeros((M,T+2))
	blk.f = np.zeros((M,T+2))
	blk.o = np.zeros((M,T+2))

	for t in range(T+1):
		if t > 0:
			blk.zb[:,t] = blk.Wz @ X[:,t] + blk.Rz @ Yhat[:,t-1] + blk.bz
			blk.z[:,t] = np.tanh(blk.zb[:,t])
			blk.ib[:,t] = blk.Wi @ X[:,t] + blk.Ri @ Yhat[:,t-1]+ blk.pi * blk.c[:,t-1] + blk.bi
			blk.i[:,t] = sig(blk.ib[:,t])
			blk.fb[:,t] = blk.Wf @ X[:,t]+ blk.Rf @ Yhat[:,t-1]+ blk.pf * blk.c[:,t-1]+blk.bf
			blk.f[:,t] = sig(blk.fb[:,t])
			blk.c[:,t] = blk.z[:,t] * blk.i[:,t] + blk.c[:,t-1] * blk.f[:,t]
			blk.ob[:,t] = blk.Wo @ X[:,t] + blk.Ro @ Yhat[:,t-1] + blk.po*blk.c[:,t]+blk.bo
			blk.o[:,t] = sig(blk.ob[:,t])
			Yhat[:,t] = np.tanh(blk.c[:,t])*blk.o[:,t]
	# for i in range(T+1):
	# 	if i > 0:
	# 		Yhat[:,i] = sig(Yhat[:,i])
	print(blk.Rz)
	return Yhat



#backprop
def backprop(X,Y,Yhat,blk,N,M,T,gamma):
	dy= np.zeros((M,T+2))
	do= np.zeros((M,T+2))
	dc= np.zeros((M,T+2))
	df= np.zeros((M,T+2))
	di= np.zeros((M,T+2))
	dz= np.zeros((M,T+2))

	deltWz = np.zeros((M,N))
	deltWi = np.zeros((M,N))
	deltWf = np.zeros((M,N))
	deltWo = np.zeros((M,N))

	deltRz = np.zeros((M,M))
	deltRi = np.zeros((M,M))
	deltRf = np.zeros((M,M))
	deltRo = np.zeros((M,M))

	deltpi = np.zeros(M)
	deltpf = np.zeros(M)
	deltpo = np.zeros(M)

	deltbz = np.zeros(M)
	deltbi = np.zeros(M)
	deltbf = np.zeros(M)
	deltbo = np.zeros(M)

	for nt in range(T):
		t = T-nt;
		Yhat[:,t] = sig(Yhat[:,t])
		dy[:,t] = Yhat[:,t]-Y[:,t] + blk.Rz.T @ dz[:,t+1] + blk.Ri.T @ di[:,t+1] + blk.Rf.T @ df[:,t+1] + blk.Ro.T @ do[:,t+1]
		do[:,t] = dy[:,t] * np.tanh(blk.c[:,t]) * dsig(blk.ob[:,t])
		dc[:,t] = dy[:,t] * blk.o[:,t] * dtanh(blk.c[:,t]) + blk.po * do[:,t] + blk.pi * di[:,t+1] + blk.pf * df[:,t+1] + dc[:,t+1] * blk.f[:,t+1]
		df[:,t] = dc[:,t] * blk.c[:,t-1] * dsig(blk.fb[:,t])
		di[:,t] = dc[:,t] * blk.z[:,t] * dsig(blk.ib[:,t])
		dz[:,t] = dc[:,t] * blk.i[:,t] * dtanh(blk.zb[:,t]) 

		deltWi += np.outer(di[:,t], X[:,t])
		deltWf += np.outer(df[:,t], X[:,t])
		deltWo += np.outer(do[:,t], X[:,t])
		deltWz += np.outer(dz[:,t], X[:,t])

		if t < T:
			deltRi += np.outer(di[:,t+1], Yhat[:,t])
			deltRf += np.outer(df[:,t+1], Yhat[:,t])
			deltRo += np.outer(do[:,t+1], Yhat[:,t])
			deltRz += np.outer(dz[:,t+1], Yhat[:,t])

			deltpi += blk.c[:,t] * di[:,t+1]
			deltpf += blk.c[:,t] * df[:,t+1]

		deltpo += blk.c[:,t] * do[:,t]

		deltbi += di[:,t]
		deltbf += df[:,t]
		deltbo += do[:,t]
		deltbz += dz[:,t]

	blk.Wi -= gamma*deltWi
	blk.Wf -= gamma*deltWf
	blk.Wo -= gamma*deltWo
	blk.Wz -= gamma*deltWz

	blk.Ri -= gamma*deltRi
	blk.Rf -= gamma*deltRf
	blk.Ro -= gamma*deltRo
	blk.Rz -= gamma*deltRz

	blk.pi -= gamma*deltpi
	blk.pf -= gamma*deltpf
	blk.po -= gamma*deltpo

	blk.bi -= gamma*deltbi
	blk.bf -= gamma*deltbf
	blk.bo -= gamma*deltbo
	blk.bz -= gamma*deltbz
	print(blk.Rz)
	

	# loss = 0;
	# for i in range(M):
	# 	for t in range(T):
	# 		# if Yhat[i,t] > 0:
	# 		# 	loss = loss - Y[i,t]*np.log(Yhat[i,t]);
	# 		if Y[i,t] == 1:
	# 			loss = loss-np.log(Yhat[i,t])
	# 		else:
	# 			loss = loss-np.log(1 - Yhat[i,t])
	loss = 0;
	for i in range(T+1):
		Yhat[:,[i]] = sig(Yhat[:,[i]])
	for i in range(M):
	 	for t in range(T):
	 		loss += (1/2)*(Yhat[i,t] - Y[i,t])*(Yhat[i,t] - Y[i,t])
	
					

	return loss,blk,Yhat[:,1:]

