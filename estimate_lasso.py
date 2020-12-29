import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from lasso_admm import lasso_admm
from solve_lasso import solve_lasso

import matplotlib.lines as mlines

def data_generation():
    n, p = 3000, 500
    ### generate true model part
    beta_true = [-2, -2, 2, 2, -2]
    x = np.random.normal(size=(n, 5))
    #print(x)
    epsilon = np.random.normal(0, 0.5, size=(n))
    y = (beta_true* x).sum(axis=1) + epsilon
    #print(y)
    ### generate redundant covariates
    true_idx = [100-1, 200-1, 300-1, 400-1, 500-1]
    X = np.random.normal(0, 1, size = (n, p))
    X[:, true_idx] = x
    
    return y, X

def problem1(y, X):
    lambdas = [0.25,0.5,0.75, 1]
    #lambdas = np.arange(2, 10, 0.5)
    for lamb in lambdas:
        print(lamb)
        primal_residuals, dual_residuals, beta = solve_lasso(X, y, lamb = lamb, num_iterations = 500, rho = 0.5)
        #print(beta.shape)
        plt.plot(primal_residuals, label='primal_residuals')
        plt.plot(dual_residuals, label='dual_residuals')
        plt.xlabel('number of iterations ')
        plt.ylabel('Euclidean norm of the primal residual and dual residual')
        plt.title(r'$\lambda = {}$'.format(lamb))
        plt.legend()
        plt.savefig('{}.png'.format(lamb))
        plt.close()

def problem2(y, X):
    betas = []
    lambdas = np.linspace(0.001, 5.0, num=20)
    print(lambdas)
    for lamb in lambdas:
        primal_residuals, dual_residuals, beta = solve_lasso(X, y, lamb = lamb, num_iterations = 500, rho = 1)
        print(beta.shape)
        betas.append(beta)
    betas = np.concatenate(betas, axis=1)
    print(betas.shape)
    ### betas (500, 20) betas.T (20, 500)
    true_idx = [100-1, 200-1, 300-1, 400-1, 500-1]
    redundant_idx = [i for i in range(500) if i not in true_idx]
    plt.plot(betas.T[:, true_idx], color='r', marker='+')
    plt.plot(betas.T[:, redundant_idx], color='gray', marker='+')
    plt.ylabel(r'$\hat{\beta}^{lasso}(\lambda)$')
    plt.xlabel(r'$\lambda$')
    blue_line = mlines.Line2D([], [], color='red', marker='+', markersize=15, 
        label='100, 200, 300, 400, 500th elements')
    plt.legend(handles=[blue_line], loc=(2/16, 2/9))
    #plt.legend()
    #plt.show()
    plt.savefig('problem2.png')
    

def main():
    y, X = data_generation()
    y = np.expand_dims(y, axis=1)
    print(X.shape, y.shape)
    #problem2(y, X)
    problem1(y, X)


if __name__ == '__main__':
    main()