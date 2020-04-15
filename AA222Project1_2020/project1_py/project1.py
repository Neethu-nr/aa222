#
# File: project1.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
        To import from another file xyz.py here, type
        import project1.xyz
        However, do not import any modules except numpy in those files.
        It's ok to import modules only in files that are
        not imported here (e.g. for your plotting code).
'''
import numpy as np
from project1_py.DescentMethods import Adam, bfgs


def optimize(f, g, x0, n, count, prob):
    """
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `g` costs twice of `f`
        count (function): takes no arguments are returns current count
        prob (str): Name of the problem. So you can use a different strategy
                 for each problem. `prob` can be `simple1`,`simple2`,`simple3`,
                 `secret1` or `secret2`
    Returns:
        x_best (np.array): best selection of variables found
    """
    minimizer = None
    # if prob == 'simple1':
    minimizer = Adam(f=f, g=g, alpha=0.3, gamma1=0.2, gamma2=0.9)
    # else:
        # minimizer = Adam(f=f, g=g)
    # minimizer = bfgs(f=f, g=g)
    x, _, _, _ = minimizer.minimize(x0, n)
    return x