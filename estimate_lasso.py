import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

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
    print(X.shape, x.shape)
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

def main():
    #np.random.seed(1)
    X, Y = data_generation()
    exit(0)

    clf = LogisticRegression(penalty='l2', tol=1e-8, max_iter=10000).fit(X, Y)
    beta_star = clf.coef_.squeeze()
    loss_beta_star = loss_function(beta_star, X, Y)
    
    inital_beta = np.random.normal(0,1,size=(10,1))
    losses, grad_norms = training(inital_beta, X, Y, iterations=10000, tol=5*1e-6)
    
    plt.plot(grad_norms)
    plt.xlabel('number of iterations')
    plt.ylabel('Euclidean norm of the gradient of the loss function (log10)')
    plt.yscale('log')
    plt.show()
    plt.close()
    
    to_proc = np.array(losses) - np.array(loss_beta_star)
    #to_proc[to_proc < 0] = 1e-5
    plt.plot(to_proc)
    plt.xlabel('number of iterations')
    plt.ylabel('difference of loss functions from optimizer')
    #plt.yscale('log')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()