import numpy as np

class DescentMethod:
	def __init__(self, f, g):
		self.count = 0
		self.maxIter = 0
		self.f_fun = f
		self.g_fun = g

	def f(self, x):
		self.count += 1
		if self.count <= self.maxIter:
			return self.f_fun(x)
		return StopIteration

	def g(self, x):
		self.count += 2
		if self.count <= self.maxIter:
			return self.g_fun(x)
		return StopIteration


	def _step(self):
		"""
	 	Updates counter
		Updates value of self.x
		returns: None
	 	"""
		raise NotImplementedError

	def minimize(self, x=0, maxIter=10, reltol_threshold=1e-3):
		"""
		wrapper to call _step
		minimizes f till reltol < threshold
		returns: x, x_history, count, reltol
		"""
		# print("new prob, maxIter:",maxIter)
		self.x = x
		self.maxIter = maxIter
		self.count = 0
		if maxIter < 1:
			return self.x, x_history, self.count, reltol
		f_x = self.f(self.x)
		eps = 1e-8

		x_history = [np.array(self.x)]
		reltol = np.inf
		reltol_threshold = abs(reltol_threshold)
		
		while abs(reltol) > reltol_threshold:
			# print("count:",self.count)
			f_prev = f_x
			try:
				self._step()
				f_x = self.f(self.x)
				reltol = (f_prev - f_x)/(f_prev + eps)
			except :
				break
			x_history.append(np.array(self.x))

		return self.x, x_history, self.count, reltol

class GradientDescent (DescentMethod):

	def __init__(self, f, g, alpha=0.001):
		DescentMethod.__init__(self,f,g)
		self.alpha = alpha

	def _step(self):
		self.x -= self.alpha * self.g(self.x)


class Adam(DescentMethod):
	"""
	parameters:
		alpha : learning rate
		gamma1 : decay parameter
		gamma2 : decay parameter
		eps : small value
		count : step counter
		v : 1st moment estimate
		s : 2nd moment estimate
	"""
	def __init__(self, f, g,  alpha=0.3, gamma1=0.2, gamma2=0.9, eps=1e-8):
		DescentMethod.__init__(self,f,g)
		self.alpha = alpha
		self.gamma1 = gamma1
		self.gamma2 =  gamma2
		self.eps = eps
		self.v = None
		self.s = None
		self.count_increment = 1


	def _step(self):
		if self.v is None:
			self.v = np.zeros(len(self.x))

		if self.s is None:
			self.s = np.zeros(len(self.x))

		g = self.g(self.x)

		# biased decaying 
		self.v = self.gamma1*self.v + (1-self.gamma1) * g
		self.s = self.gamma2*self.s + (1-self.gamma2) * np.multiply(g,g)

		# corrected decaying
		v_hat = self.v / (1 - pow(self.gamma1,self.count))
		s_hat = self.s / (1 - pow(self.gamma2,self.count))
		self.x -= self.alpha * np.divide(v_hat, np.sqrt(s_hat) + self.eps)

class bfgs(DescentMethod):
	def __init__(self, f, g):		
		DescentMethod.__init__(self,f,g)
		self.Q = None

	def _step(self):
		if self.Q is None:
			self.Q = np.eye(len(self.x))
		g = self.g(self.x)

		x_prime = self.approx_line_search(-np.dot(self.Q, g))
		g_prime = self.g(x_prime)
		delta = x_prime - self.x
		y = g_prime - g
		y_t = y.T
		delta_t = delta.T
		yQy = np.dot(np.dot(y_t,self.Q),y)
		delta_term = np.dot(delta, delta_t) / np.dot(delta_t, y) 
		self.Q -= np.dot(y, y_t) * np.dot(self.Q, self.Q) / yQy + delta_term 
		self.x = x_prime 

	def backtracking_line_search(self, d, alpha=1, p=0.75, beta=1e-4):
		"""
		Approximate line search to find a suitable step size with a
		small number of evaluations. The condition for sufficient decrease is
		used here. It requires that the step size cause a sufficient edecrease
		in the objective function value. 

		Args:	    
		    d (np.array): Descent direction 
		    alpha (float): Maximum step size
		    p (float):Reduction factor 
		    beta (float): First Wolfe condition parameter
		"""
		y, g = self.f(self.x), self.g(self.x)
		while self.f(self.x + alpha*d) > y + beta*alpha*np.dot(g, d) :
			alpha *= p
		return alpha

	def approx_line_search(self, d):
	    alpha = self.backtracking_line_search(d)
	    return self.x + alpha * d 
