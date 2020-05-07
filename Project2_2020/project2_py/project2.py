#
# File: project2.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
		To import from another file xyz.py here, type
		import project2_py.xyz
		However, do not import any modules except numpy in those files.
		It's ok to import modules only in files that are
		not imported here (e.g. for your plotting code).
'''
import numpy as np
from project2_py.ConstrainedOptimizers import *

def optimize(f, g, c, x0, n, count, prob):
	"""
	Args:
		f (function): Function to be optimized
		g (function): Gradient function for `f`
		c (function): Function evaluating constraints
		x0 (np.array): Initial position to start from
		n (int): Number of evaluations allowed. Remember `f` and `c` cost 1 and `g` costs 2
		count (function): takes no arguments are returns current count
		prob (str): Name of the problem. So you can use a different strategy 
				 for each problem. `prob` can be `simple1`,`simple2`,`simple3`,
				 `secret1` or `secret2`
	Returns:
		x_best (np.array): best selection of variables found
	"""
	optimizer = constrainedOptimizer(f, g, c, n, count, penalty="combined")
	x_best = optimizer.nelder_mead(x0)

	return x_best