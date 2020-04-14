from project1_py.project1 import optimize
from project1_py import helpers
import matplotlib.pyplot as plt

if __name__ == '__main__':
    	
    '''
    Plots results of optimize
    '''

    for problem in [Simple1, Simple2, Simple3]:

        print('Working on %s...' % p.prob)
        fvals_opt = []
        for seed in tqdm(range(3)):
	        p = problem()
            np.random.seed(seed)
            x0 = p.x0()
            p.nolimit()
            xb = optimize(p.f, p.g, x0, p.n, p.count, p.prob)
            p._reset()
            fvals_opt.append(p.f(xb))
        
        if np.any(np.isnan(fvals_opt)):
            print('Warning: NaN returned by optimizer. Leaderboard score will be 0.')
            fvals_opt = np.where(np.isnan(fvals_opt), np.inf, fvals_opt)

        better = np.array(fvals_random) > np.array(fvals_opt)
