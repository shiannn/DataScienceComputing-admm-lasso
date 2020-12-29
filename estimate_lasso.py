import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from lasso_admm import lasso_admm
from solve_lasso import solve_lasso

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

def training(beta, X, Y, iterations = 10000, tol=5*1e-6):
    iteration = 0
    losses = []
    grad_norms = []
    loss = np.inf
    while(iteration<iterations and loss > tol):
        loss = loss_function(beta, X, Y)
        print('loss', loss)
        grad = gradient(beta, X, Y)
        #print('grad norm', )
        grad_norm = np.linalg.norm(grad, ord=2)
        
        grad = np.expand_dims(grad, 1)
        #print('grad', grad.shape)
        beta = beta + 0.12* grad
        #print('beta', beta.shape)
        iteration += 1
        losses.append(loss)
        grad_norms.append(grad_norm)
    
    return losses, grad_norms

def problem1(y, X):
    primal_residuals, dual_residuals, beta = solve_lasso(X, y, lamb = 0.3, num_iterations = 500, rho = 1)
    print(beta.shape)
    plt.plot(primal_residuals, label='primal_residuals')
    plt.plot(dual_residuals, label='dual_residuals')
    plt.legend()
    plt.show()

def problem2(y, X):
    betas = []
    lambdas = np.linspace(0.001, 5.0, num=20)
    print(lambdas)
    for lamb in lambdas:
        primal_residuals, dual_residuals, beta = solve_lasso(X, y, lamb = lamb, num_iterations = 20, rho = 1)
        print(beta.shape)
        betas.append(beta)
    betas = np.concatenate(betas, axis=1)
    print(betas.shape)
    ### betas (500, 20) betas.T (20, 500)
    true_idx = [100-1, 200-1, 300-1, 400-1, 500-1]
    redundant_idx = [i for i in range(500) if i not in true_idx]
    plt.plot(betas.T[:, true_idx], color='r', marker='+')
    plt.plot(betas.T[:, redundant_idx], color='gray', marker='+')
    #plt.legend()
    plt.show()
    

def main():
    #np.random.seed(1)
    y, X = data_generation()
    y = np.expand_dims(y, axis=1)
    print(X.shape, y.shape)
    #problem2(y, X)
    problem1(y, X)
    
    """
    z, h = lasso_admm(X,y,alpha=1.,rho=1.,rel_par=1.,QUIET=False, MAX_ITER=500,ABSTOL=5e-3,RELTOL=5e-3)
    #print(h['eps_dual'])
    #print(h['eps_pri'])
    dual_residual = np.array(h['eps_dual'])
    prime_residual = np.array(h['eps_pri'])

    dual_residual = dual_residual[dual_residual!=0]
    prime_residual = prime_residual[prime_residual!=0]
    print(dual_residual)
    print(prime_residual)

    plt.plot(dual_residual)
    plt.plot(prime_residual)
    plt.show()
    """


if __name__ == '__main__':
    main()