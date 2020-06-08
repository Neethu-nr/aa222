# combine genes in different ways to get children

import numpy as np

class CrossoverMethod:
	def __init__(self, gene_length, lam=0.5):
		self.gene_length = gene_length
		self.lam = lam
		self.child = [0]*gene_length

	def singlePoint(self,a,b):
			i = np.random.randint(self.gene_length)
			return a[:i] + b[i:]

	def twoPoint(self,a,b):
		i = np.random.randint(self.gene_length)
		j = np.random.randint(self.gene_length)
		if i > j:
			i, j = j, i
		return a[:i] + b[i:j] + a[j:]

	def uniform(self,a,b):
		child = list(a)
		prob = np.random.normal(0.5,0.5,self.gene_length)
		child = np.where(prob>0.5, a,b)
		return list(child)

	def interpolation(self,a,b):
		for i in range(self.gene_length):
			self.child[i] = (1-self.lam) * a[i] + self.lam*b[i] 
		return self.child