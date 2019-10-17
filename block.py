import numpy as np

class block:
	def __init__(self,N,M,T):
		self.Wz = np.random.normal(.1, .1, (M,N))
		self.Wi = np.random.normal(.1, .1, (M,N))
		self.Wf = np.random.normal(.1, .1, (M,N))
		self.Wo = np.random.normal(.1, .1, (M,N))

		self.Rz = np.random.normal(.1, .1, (M,M))
		self.Ri = np.random.normal(.1, .1, (M,M))
		self.Rf = np.random.normal(.1, .1, (M,M))
		self.Ro = np.random.normal(.1, .1, (M,M))

		self.pi = np.zeros(M)
		self.pf = np.zeros(M)
		self.po = np.zeros(M)

		self.bz = np.zeros(M)
		self.bi = np.zeros(M)
		self.bf = np.zeros(M)
		self.bo = np.zeros(M)

		self.zb = np.zeros((M,T+2))
		self.ib = np.zeros((M,T+2))
		self.fb = np.zeros((M,T+2))
		self.ob = np.zeros((M,T+2))
		self.c = np.zeros((M,T+2))
		self.z = np.zeros((M,T+2))
		self.i = np.zeros((M,T+2))
		self.f = np.zeros((M,T+2))
		self.o = np.zeros((M,T+2))
		