import numpy as np
"""
def coordinate_descent(z, x, nu, rho, lamb):
    e_grad = x + nu*1.0/rho
    rho = 1
    # Regularization term gradient
    # This will have a subgradient, with values as -lambda/rho, lambda/rho OR 0

    # print("prev",z)
    z_t = np.zeros_like(z)

    filter_less = -(1.0*lamb/rho)*(z<0)
    # print("less",filter_less)
    filter_greater = (1.0*lamb/rho)*(z>0)
    # print("gt",filter_greater)

    z_t = e_grad - filter_less - filter_greater
    # print(z_t)
    return(z_t)
"""
def soft_thresholding(beta, omega, rho, lamb):
    input_term = beta + omega
    t = lamb / rho
    #print(t)
    #np.where(input_term > t, input_term - t, )
    input_term = np.select(
        [input_term > t, (input_term <= t) & (input_term >= -t), input_term < -t],
        [input_term - t, 0, input_term + t]
    )
    #print(input_term)

    return input_term

def solve_lasso(X, y, lamb = 0.25, num_iterations = 10, rho = 1):
    n, p = X.shape
    print(n, p)
    #beta = np.random.randn(p, 1)
    # z_t = np.random.randn(d, 1)
    # X_t = np.zeros((d,1))
    beta = np.zeros((p,1))
    alpha = np.zeros((p,1))
    omega = np.zeros((p,1))
    #rho = 1
    #num_iterations = 10
    #lamb = 0.25
    val = 0.5*np.linalg.norm(X.dot(beta) - y, ord=2)**2 + lamb*np.linalg.norm(beta, ord=1)
    print(val)

    primal_residuals = []
    dual_residuals = []
    #betas = []
    for iteration in range(num_iterations):
        # STEP 1: Calculate beta
        # This has a closed form solution
        term1 = np.linalg.inv(X.T.dot(X) + rho* np.identity(X.shape[1]))
        term2 = X.T.dot(y) + rho*(alpha -  omega)
        beta = term1.dot(term2)

        # STEP 2: Calculate z_t
        alpha_old = alpha
        #alpha = coordinate_descent(alpha, beta, omega, rho, lamb)
        alpha = soft_thresholding(beta, omega, rho, lamb)
        print(alpha.shape)

        # STEP 3: Update nu_t
        omega = omega + rho*(beta - alpha)
        val = 0.5*np.linalg.norm(X.dot(beta) - y, ord=2)**2 + lamb*np.linalg.norm(beta, ord=1)
        print(val)

        primal_residual = beta + alpha
        primal_residual = np.linalg.norm(primal_residual, ord=2)
        dual_residual = rho*(alpha - alpha_old)
        dual_residual = np.linalg.norm(dual_residual, ord=2)

        primal_residuals.append(primal_residual)
        dual_residuals.append(dual_residual)
        #betas.append(beta)
    
    return primal_residuals, dual_residuals, beta