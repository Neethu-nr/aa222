# mutate a gene
import numpy as np

class MutationMethod:
	def __init__(self, gene_length, mu=0, sigma=1):
		self.gene_length, self.mu, self.sigma = gene_length, mu, sigma
		self.lam = 1/gene_length

	def BitwiseMutation(self, a):
		# for binary genes
		prob  = np.random.normal(0.5, 1, self.gene_length)
		return [a[i] if prob[i]>self.lam else not a[i] for i in range(self.gene_length)]

	def GaussianMutation (self, a):
		# for real valued genes
		noise  = np.random.normal(self.mu, self.sigma, self.gene_length)
		return [a[i] + noise[i] for i in range(self.gene_length)]

	def BitwiseGaussian(self,a):
		# import pdb;pdb.set_trace()
		prob  = np.random.normal(0.5, 1, self.gene_length)
		noise  = np.random.normal(self.mu, self.sigma, self.gene_length)
		return [a[i] if prob[i]>self.lam else a[i]+noise[i] for i in range(self.gene_length)]
