import numpy as np

def backtracking_line_search(fun, grad, x, d, maxIter,alpha=1.5, p=0.5, beta=1e-4):
    """
    Approximate line search to find a suitable step size with a
    small number of evaluations. The condition for sufficient decrease is
    used here. It requires that the step size cause a sufficient edecrease
    in the objective function value. 

    Args:
    
        fun (function): Function to be optimized
        grad (function): Gradient function for `f`
        x (np.array): Initial position to start from
        d (np.array): Descent direction 
        alpha (float): Maximum step size
        p (float):Reduction factor 
        beta (float): First Wolfe condition parameter
    """
    y, g = fun(x), grad(x)
    count = 3
    print("maxiter:" ,maxIter)
    while count < maxIter-1 and fun(x + alpha*d) > y + beta*alpha*np.dot(g,d) :
        alpha *= p
        count += 1
    return alpha, count

def approx_line_search(f,g,x,d, maxIter):
    alpha, count = backtracking_line_search(f,g,x,d, maxIter)
    return x + alpha * d, count 

def strong_backtracking(fun, grad, x, d, alpha=1, beta=1e-4, sigma=0.1):
    """
    Approximate line search for satisfying the strong Wolfe conditions.
    The algorithm’s bracket phase first brackets an interval containing 
    a step size that satisfies the strong Wolfe conditions. It then 
    reduces this bracketed interval in the zoom phase until a suitable 
    step size is found. We interpolate with bisection, but other 
    schemes can be used.

    Args:
    
        fun (function): Function to be optimized
        grad (function): Gradient function for `f`
        x (np.array): Initial position to start from
        d (np.array): Descent direction 
        alpha (float): Maximum step size
        beta, sigma (floats) : Wolfe condition parameters
    """
    y0, g0, y_prev, alpha_prev = fun(x), np.dot(grad(x),d), None, 0
    alpha_lo, alpha_hi = None, None
    count = 3
    # bracket phase
    while True:
        y = fun(x + alpha*d)
        count += 1
        if (y > (y0 + beta*alpha*g0)) or ((y_prev is not None) and y >= y_prev):
            alpha_lo, alpha_hi = alpha_prev, alpha
            break

        g = np.dot(grad(x + α*d),d)
        count += 2
        if abs(g) <= -sigma*g0:
            return alpha, count
        elif g >= 0:
            alpha_lo, alpha_hi = alpha, alpha_prev   
            break
        y_prev, alpha_prev, alpha = y, alpha, 2*alpha
    
    # zoom phase
    y_lo = fun(x + alpha_lo*d)
    count += 1

    while True:
        alpha = (alpha_lo + alpha_hi)/2
        y = fun(x + alpha*d)
        count += 1
        if (y > y0 + beta*alpha*g0) or y >= y_lo:
            alpha_hi = alpha
        else:
            g = np.dot(grad(x + α*d),d)
            count += 2
            if abs(g) <= -sigma*g0:
                return alpha, count
            elif g*(alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
        
            alpha_lo = alpha
