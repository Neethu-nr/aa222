# mutate a gene
import numpy as np

class MutationMethod:
	def __init__(self, gene_length, mu=0, sigma=1):
		self.gene_length, self.mu, self.sigma = gene_length, mu, sigma
		self.lam = 1/gene_length

	def BitwiseMutation(self, a):
		# for binary genes
		prob  = np.random.normal(0.5, 1, self.gene_length)
		return [g if prob>self.lam else not g for g in a]

	def GaussianMutation (self, a):
		# for real valued genes
		noise  = np.random.normal(self.mu, self.sigma, self.gene_length)
		return [a[i] + noise[i] for i in range(self.gene_length)]