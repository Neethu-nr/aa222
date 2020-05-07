#
# File: helpers.py
#

# this file defines the optimization problems and test

from tqdm import tqdm
import numpy as np
from project2_py.ConstrainedOptimizers import *

class OptimizationProblem:

	@property
	def xdim(self):
		# dimension of x
		return self._xdim

	@property
	def prob(self):
		# problem name
		return self._prob
	
	@property
	def n(self):
		# number of allowed evaluations
		return self._n
	
	def _reset(self):
		self._ctr = 0

	def count(self):
		return self._ctr

	def nolimit(self):
		# sets n to inf, useful for plotting/debugging
		self._n = np.inf
		
	def x0(self):
		'''
		Returns:
			x0 (np.array): (xdim,) randomly initialized x
		'''
		return np.random.randn(self.xdim)

	def f(self, x):
		'''Evaluate f
		Args:
			x (np.array): input
		Returns:
			f (float): evaluation
		'''
		assert x.ndim == 1

		self._ctr += 1

		return self._wrapped_f(x)
	
	def _wrapped_f(self, x):
		raise NotImplementedError

	def g(self, x):
		'''Evaluate jacobian of f
		Args:
			x (np.array): input
		Returns:
			jac (np.array): jacobian of f wrt x
		'''
		assert x.ndim == 1

		self._ctr += 2

		return self._wrapped_g(x)
class Rosenbrock(OptimizationProblem):
    '''
    Rosenbrock's Function
    '''
    
    def __init__(self):
        self._xdim = 2
        self._prob = 'simple1'
        self._n = 20
        self._reset()

    def x0(self):
        return np.clip(np.random.randn(self.xdim), -3, 3)

    def _wrapped_f(self, x):
        return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

    def _wrapped_g(self, x):
        return np.array([
            2*(-1 + x[0] + 200*x[0]**3 - 200*x[0]*x[1]),
            200*(-x[0]**2 + x[1])
                ])

class ConstrainedOptimizationProblem(OptimizationProblem):

	@property
	def cdim(self):
		# number of constraints
		return self._cdim

	@property
	def nc(self):
		# number of allowed constraint evals
		return self._nc

	def _reset(self):
		self._ctr = 0

	def count(self):
		return self._ctr

	def nolimit(self):
		# sets n to inf, useful for plotting/debugging
		self._n = np.inf

	def c(self, x):
		'''Evaluate constraints
		Args:
			x (np.array): input
		Returns:
			c (np.array): (cdim,) evaluation of constraints
		'''
		assert x.ndim == 1
		
		self._ctr += 1

		return self._wrapped_c(x)

	def _wrapped_c(self, x):
		raise NotImplementedError


class Simple1(ConstrainedOptimizationProblem):
	
	def __init__(self):
		self._xdim = 2
		self._cdim = 2
		self._prob = 'simple1'
		self._n = 2000
		self._reset()

	def x0(self):
		return np.random.rand(self._xdim) * 2.0

	def _wrapped_f(self, x):
		return -x[0] * x[1]  + 2.0 / (3.0 * np.sqrt(3.0))

	def _wrapped_g(self, x):
		return np.array([
			-x[1],
			-x[0],
				])

	def _wrapped_c(self,x):
		return np.array([
			x[0] + x[1]**2 - 1,
			-x[0] - x[1]
			])



class Simple2(ConstrainedOptimizationProblem):

	def __init__(self):
		self._xdim = 2
		self._cdim = 2
		self._prob = 'simple2'
		self._n = 2000
		self._reset()

	def x0(self):
		return np.random.rand(self._xdim) * 2.0 - 1.0

	def _wrapped_f(self, x):
		return 100 * (x[1] - x[0]**2)**2 + (1-x[0])**2

	def _wrapped_g(self, x):
		return np.array([
			2*(-1 + x[0] + 200*x[0]**3 - 200*x[0]*x[1]),
			200*(-x[0]**2 + x[1])
				])

	def _wrapped_c(self,x):
		return np.array([
			(x[0]-1)**3 - x[1] + 1,
			x[0] + x[1] - 2,
			])


class Simple3(ConstrainedOptimizationProblem):

	def __init__(self):
		self._xdim = 3
		self._cdim = 1
		self._prob = 'simple3'
		self._n = 2000
		self._reset()

	def x0(self):
		b = 2.0 * np.array([1.0, -1.0, 0.0])
		a = -2.0 * np.array([1.0, -1.0, 0.0])
		return np.random.rand(3) * (b-a) + a

	def _wrapped_f(self, x):
		return x[0] - 2*x[1] + x[2] + np.sqrt(6.0)

	def _wrapped_g(self, x):
		return np.array([1., -2., 1.])

	def _wrapped_c(self, x):
		return np.array([x[0]**2 + x[1]**2 + x[2]**2 - 1.])



def test_optimize(optimize):
	'''
	Tests optimize to ensure it passes
	Args:
		optimize (function): function optimizing a given problem
	'''
	def test1(verbose=True):
		if verbose:
			print("\ntest1: creating optimizer obj")
		p = Simple1()
		optimizer = constrainedOptimizer(p.f, p.g, p.c, p.n, p.count, penalty=None)
		x0 = p.x0()
		if verbose:
			print("passed")
		return p, optimizer, x0

	def test2():
		print("\ntest2: calling f max times from it")
		p, optimizer, x0 = test1(False)
		for _ in tqdm(range(p.n+30)):
			try:
				optimizer.f(x0)
			except:
				break
		print(p.n, p.count())
		assert (p.n == p.count())

		print("\ntest2: calling g max times from it")
		p._reset()	
		for _ in tqdm(range(p.n+30)):
			try:
				optimizer.g(x0)
			except:
				break
		print(p.n, p.count())
		assert (p.n == p.count())

		print("\ntest2: calling g max times from it")	
		p._reset()	
		for _ in tqdm(range(p.n+30)):
			try:
				optimizer.c(x0)
			except:
				break
		print(p.n, p.count())
		assert p.n == p.count() or p.n-1 == p.count()
		print("passed")

	def test3():
		print("\ntest3: penalty functions")
		p, optimizer, x0 = test1(False)
		constraints = p.c(x0)

		count = optimizer.penalty_count(x0)
		print("constraints:",constraints, "\n#count of c> 0:",count)
		sq = optimizer.penalty_square(x0)
		print("constraints:",constraints, "\nsquared sum of c> 0:",sq)
		print("check if passed")

	def test4():
		print("\ntest4: init_simples")
		p, optimizer, x0 = test1(False)
		S = optimizer.init_simplex(x0)
		print("x0:", x0.shape, "\n:", x0)
		print("S:", S.shape, "\n", S)
		print("passed")

	def test5():
		print("\ntest5: nelder mead on Rosenbrock")
		p = Rosenbrock()
		p.nolimit()
		p_c = Simple1()
		optimizer = constrainedOptimizer(p.f, p.g, p_c.c, p.n, p.count, penalty =None, verbose=True)
		x0 = p.x0()
		x = optimizer.nelder_mead(x0)
		print("passed")

	def test6():
		print("\ntest5: nelder mead with penalty on Simple1")
		p, _, x0 = test1(False)
		optimizer = constrainedOptimizer(p.f, p.g, p.c, p.n, p.count, penalty ="square", verbose=True)
		x = optimizer.nelder_mead(x0)
		print("passed")

	def final_test():
		print("Running final test")
		for test in [Simple1, Simple2, Simple3]:

			p = test()
			print('Testing on %s...' % p.prob)

			solution_feasible = []
			any_count_exceeded = False
			for seed in tqdm(range(500)):
				p = test()
				np.random.seed(seed)
				x0 = p.x0()
				xb = optimize(p.f, p.g, p.c, x0, p.n, p.count, p.prob)
				if p.count() > p.n:
					any_count_exceeded = True
					break
				p._reset()
				solution_feasible.append(np.all(p.c(xb) <= 0.0))

			if any_count_exceeded:
				print('Failed %s. Count exceeded.'%p.prob)
				continue

			# to pass, optimize must return a feasible point >=95% of the time.

			numfeas = np.sum(solution_feasible)
			if numfeas >= 0.95*500:
				print('Pass: optimize returns a feasible solution on %d/%d random seeds.' % (numfeas,500))
			else:
				print('Fail: optimize returns a feasible solution on %d/%d random seeds.' % (numfeas,500))

		return
	
	# test1()
	# test2()
	# test3()
	# test4()
	# test5()
	# test6()
	final_test()
