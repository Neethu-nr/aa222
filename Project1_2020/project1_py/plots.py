from helpers import *
from DescentMethods import Adam, bfgs
import matplotlib.pyplot as plt

if __name__ == '__main__':
    	
    '''
    Plots results of optimize
    '''
    
    # Rosenbrock’s function with the objective contours and the path taken algorithm
    fig = plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 

    start, stop, n_values = -2, 2, 1000

    x_vals = np.linspace(start, stop, n_values)
    y_vals = np.linspace(start, stop, n_values)
    X, Y = np.meshgrid(x_vals, y_vals)

    problem  = Simple1()
    problem.nolimit()
    Z = [X.reshape(1,-1), Y.reshape(1,-1)]
    Z = problem.f(Z)
    X.reshape(n_values, n_values)
    Y.reshape(n_values, n_values)
    Z = Z.reshape(n_values, n_values)
    # print(X.shape,Y.shape,Z.shape)
    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)

    ax.set_title('Rosenbrock’s function and the path taken by adam optimizer')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    seed = [100,20,30]
    color = ['r','g','b']
    for i in tqdm(range(3)):
        np.random.seed(seed[i])
        x0 = problem.x0()
        minimizer = Adam(f=problem.f, g=problem.g)
        _, x_hist, _, _ = minimizer.minimize(x0, problem.n)
        x_hist = np.array(x_hist).reshape(-1,2)
        plt.plot(x_hist[:,0],x_hist[:,1], color[i])
        problem._reset()
    
    plt.show()


    # convergence plots
    seed= [10,25,10]
    i = -1
    for problem in [Simple1, Simple2, Simple3]:
        i += 1
        p = problem()
        np.random.seed(seed[i])
        x0 = p.x0()
        minimizer = Adam(f=p.f, g=p.g)
        _, x_hist, _, _ = minimizer.minimize(x0, p.n)
        x_hist = np.array(x_hist).reshape(p._xdim,-1)
        p._reset()
        f_hist = p.f(x_hist)
        print(f_hist.shape)
        plt.plot(f_hist, color[i])

    plt.legend(['Rosenbrock', 'Himmelblau','Powell'])
    plt.title('Convergence plot')
    plt.xlabel('Function value')
    plt.ylabel('Number of evaluations')
    plt.show()


    # # convergence plots
    # seed= [10,25,10]
    # i = -1
    # for problem in [Simple1, Simple2, Simple3]:
    #     i += 1
    #     p = problem()
    #     np.random.seed(seed[i])
    #     x0 = p.x0()
    #     minimizer = Adam(f=p.f, g=p.g)
    #     _, x_hist, _, _ = minimizer.minimize(x0, p.n)
    #     x_hist = np.array(x_hist).reshape(p._xdim,-1)
    #     p._reset()
    #     f_hist = p.f(x_hist).reshape(-1,1)
    #     print(f_hist.shape, f_hist)
    #     rel = np.zeros((f_hist.shape[0],1))
    #     for j in range(1, f_hist.shape[0]):
    #         rel[j] = abs((f_hist[j]-f_hist[j-1])/(f_hist[j-1] + 1e-8))

    #     plt.plot(rel, color[i])
    # ax.legend(['Rosenbrock', 'Himmelblau','Powell'])
    # ax.set_title('Convergence plot')
    # ax.set_xlabel('Function value')
    # ax.set_ylabel('Number of evaluations')
    # plt.show()

