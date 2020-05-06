from project2.py import Adam
import numpy as np

class constrainedOptimizer(object):
	"""
	Different methods for optimizing constrained problems.
	"""
	def __init__(self, ):
		
def penalty_count(constraint):
	return np.sum(constraint() > 0)

def penalty_square(constraint):
	constraint_val = constraint()
	constraint_val[constraint_val<0] = 0
	return np.sum(constraint_val ** 2)

def init_simplex(dim):
	"""
	Creates simplex in n dimension
	"""
	simplex = [i*[0]+[dim]+(dim+~i)*[0]for i in range(dim)]
	return np.array((simplex + [dim*[1+(dim+1)**.5]]))

def penalty_method(f, p, x, k_max, rho=1, gamma=2):

	for k in range(1,k_max):
		x, _, _, _ = minimizer.minimize(f(x) + ρ*p(x), n)

		rho *= gamma
		if p(x) == 0:
			return x
	return x

def nelder_mead(f, x0, eps, alpha=1.0, beta=2.0, gamma=0.5):
	dim = len(x0)
	S = init_simplex(dim) + 
	delta = np.inf
	y_arr = np.array([f(S[i]) for i in range(dim)])
	while delta > eps:
		idx = y_arr.srgsort()
		S, y_arr = S[idx], y_arr[idx]
		xl, yl = S[0], y_arr[0]				# lowest
		xh, yh = S[end], y_arr[end]			# highest
		xs, ys = S[end-1], y_arr[end-1]		# second-highest
		xm = np.mean(S[1:end-1], axis=0)	# centroid
		xr = xm + alpha*(xm - xh)			# reflection point
		yr = f(xr)
	
		if yr < yl :
			xe = xm + beta*(xr-xm) 			# expansion point
			ye = f(xe)
			S[end], y_arr[end] = (ye < yr) ? (xe, ye) : (xr, yr)
		elif yr > ys :
			if yr <= yh :
				xh, yh, S[end], y_arr[end] = xr, yr, xr, yr
			xc = xm + gamma*(xh - xm) 		# contraction point
			yc = f(xc)
			if yc > yh:
				for i in range(1 : len(y_arr))
					S[i] = (S[i] + xl)/2
					y_arr[i] = f(S[i])	
			else:
				S[end], y_arr[end] = xc, yc
				
		else:
			S[end], y_arr[end] = xr, yr
			end
		delta = np.std(y_arr)

	return S[np.argmin(y_arr)]

def augmented_lagrange_method(f, h, x, k_max, rho=1, gamma=2)
	lam = zeros(length(h(x)))
	for k in range(1,k_max):
		p = f(x) + rho/2*sum(h(x).^2) - lam * h(x)
		x, _, _, _ = minimizer.minimize(f(x) + ρ*p(x), n)
		rho *= gamma
		lam -= rho*h(x)
	return x
