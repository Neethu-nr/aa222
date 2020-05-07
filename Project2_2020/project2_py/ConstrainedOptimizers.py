# from project2.py import Adam
import numpy as np

class constrainedOptimizer(object):
	"""
	Different methods for optimizing constrained problems.
	"""
	def __init__(self, f, g, c, n, count, penalty="square", verbose=False):
		self.count = count
		self.maxIter = n
		self.f_fun = f
		self.g_fun = g
		self.c_fun = c
		self.p = None
		if penalty == "square":
			self.p =self.penalty_square
		elif penalty == "count":
			self.p = self.penalty_count
		elif penalty is None:
			pass
		elif penalty == "combined":
			self.p = self.penalty_combined
			self.p1 = 50
			self.p2 = 750
			#(2,5) pass, f,pass
		else:
			raise NotImplementedError
		self.verbose = verbose

	def f(self, x):
		if self.count() < self.maxIter:
			if self.p is not None:
				return self.f_fun(x) + self.p(x)
			else: 
				return self.f_fun(x) 
		return StopIteration

	def g(self, x):
		if self.count() < self.maxIter:
			return self.g_fun(x)
		return StopIteration

	def c(self, x):
		if self.count() < self.maxIter-1:
			return self.c_fun(x)
		return StopIteration
		
	def penalty_count(self, x):
		"""Count of active constraints"""
		return np.sum(self.c(x) > 0)

	def penalty_square(self, x):
		""" Squared sum of active constrainrs"""
		constraint_val = self.c(x)
		constraint_val[constraint_val<0] = 0
		return np.sum(constraint_val ** 2)
		
	def penalty_combined(self, x):
		"""
		Mix of count and square penalty
		self.penalty_count() and self.penalty_square()
		are not used to reduce calls to self.c()
		"""
		constraint_val = self.c(x)
		constraint_val[constraint_val<0] = 0
		p_sq = np.sum(constraint_val ** 2)
		p_count = np.sum(self.c(x) > 0)
		return self.p1 * p_count + self.p2 * p_sq

	def init_simplex(self, x0):
		"""
		Creates simplex in n dimension
		"""
		dim = len(x0)
		simplex = [i*[0]+[dim]+(dim+~i)*[0]for i in range(dim)]
		return np.array((simplex + [dim*[1+(dim+1)**.5]])) + x0

	def nelder_mead(self, x0, eps=1e-4, alpha=1.0, beta=2.0, gamma=0.5):
		S = self.init_simplex(x0)  
		delta = np.inf
		y_arr = np.array([self.f(s) for s in S])
	
		if self.verbose:
			print("Initial:\nS:", S, "\ny_arr:",y_arr)
			y_history = []
		while delta > eps:
			try:
				# Sort the simplex entries
				idx = y_arr.argsort()
				S, y_arr = S[idx], y_arr[idx]
				if self.verbose:
					y_history.append(y_arr[0])

				# Compute the reflection point
				xl, yl	 = S[0], y_arr[0]			# lowest
				xh, yh 	 = S[-1], y_arr[-1]			# highest
				xs, ys 	 = S[-2], y_arr[-2]			# second-highest
				xm = np.mean(S[0:-2], axis=0)		# centroid
				xr = xm + alpha*(xm - xh)			# reflection point
				yr = self.f(xr)
				
				# update
				if yr < yl :
					# Compute the expansion point
					xe = xm + beta*(xr-xm) 		# expansion point
					ye = self.f(xe)
					S[-1], y_arr[-1] = (xe, ye) if (ye < yr) else (xr, yr)
				elif yr >= ys :
					if yr < yh :
						xh, yh, S[-1], y_arr[-1] = xr, yr, xr, yr
					xc = xm + gamma*(xh - xm) 		# contraction point
					yc = self.f(xc)
					if yc > yh:
						for i in range(1, len(y_arr)):
							S[i] = (S[i] + xl)/2
							y_arr[i] = self.f(S[i])	
					else:
						S[-1], y_arr[-1] = xc, yc
						
				else:
					S[-1], y_arr[-1] = xr, yr
				delta = np.std(y_arr)
			except :
				break

		if self.verbose:
			import matplotlib.pyplot as plt
			plt.figure()
			plt.plot(y_history)
			plt.xlabel('iterations')
			plt.ylabel('y_min')
			plt.show()

			idx = y_arr.argsort()
			S, y_arr = S[idx], y_arr[idx]
			print("Final\nS:", S, "\ny_arr:",y_arr)
			# print("y calculated:", np.array([self.f(s) for s in S]))
			print("solution:",S[np.argmin(y_arr)])
			print("iterations:",self.count())
		return S[np.argmin(y_arr)]


def penalty_method(f, p, x, k_max, rho=1, gamma=2):

	for k in range(1,k_max):
		x, _, _, _ = minimizer.minimize(f(x) + ρ*p(x), n)

		rho *= gamma
		if p(x) == 0:
			return x
	return x


def augmented_lagrange_method(f, h, x, k_max, rho=1, gamma=2):
	lam = zeros(length(h(x)))
	for k in range(1,k_max):
		# print(h(x))
		p = f(x) + rho/2*sum(h(x)**2) - lam * h(x)
		x, _, _, _ = minimizer.minimize(f(x) + ρ*p(x), n)
		rho *= gamma
		lam -= rho*h(x)
	return x
