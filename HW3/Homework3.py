import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Heart Attack Clean.csv")

Y = df["thalach"].to_numpy(dtype=float)
chol = df["chol"].to_numpy(dtype=float)

# First column corresponds to beta_0; second corresponds to beta_1.
X = np.column_stack((np.ones(len(chol)), chol))


# B1
def calc_gradient_ols(Y, X, beta_0, beta_1):
    betas = np.array([beta_0, beta_1], dtype=float)
    y_hat = X @ betas
    residuals = Y - y_hat

    # -(2/n) X^T(Y - Y_hat)
    gradients = -(2 / len(Y)) * X.T @ residuals
    return gradients


# B2
def calc_gradient_lasso(Y, X, beta_0, beta_1, lambda_):
    gradients = calc_gradient_ols(Y, X, beta_0, beta_1)

    # Do not penalize the intercept beta_0.
    gradients[1] += lambda_ * np.sign(beta_1)

    return gradients


# B3
def gradient_descent(
    Y, X, eta, gradient_function, lambda_=None,
    tolerance=0.1, max_iterations=1000
):
    beta_0 = 1.0
    beta_1 = 0.0
    num_iterations = 0

    while num_iterations < max_iterations:
        if lambda_ is None:
            gradients = gradient_function(
                Y, X, beta_0, beta_1
            )
        else:
            gradients = gradient_function(
                Y, X, beta_0, beta_1, lambda_
            )

        # "or" is equivalent to checking whether any gradient
        # is still larger than the tolerance.
        if not np.any(np.abs(gradients) > tolerance):
            break

        beta_0 -= eta * gradients[0]
        beta_1 -= eta * gradients[1]
        num_iterations += 1

    return np.array([beta_0, beta_1]), num_iterations


# B1 verification
initial_ols_gradient = calc_gradient_ols(
    Y, X, beta_0=1, beta_1=0
)
print("Initial OLS gradient:", initial_ols_gradient)


# B2 verification
initial_lasso_gradient = calc_gradient_lasso(
    Y, X, beta_0=1, beta_1=0, lambda_=0.5
)
print("Initial Lasso gradient:", initial_lasso_gradient)


# B4
ols_betas, ols_iterations = gradient_descent(
    Y=Y,
    X=X,
    eta=0.01,
    gradient_function=calc_gradient_ols
)

print("OLS betas:", ols_betas)
print("OLS iterations:", ols_iterations)


# B5
lasso_betas, lasso_iterations = gradient_descent(
    Y=Y,
    X=X,
    eta=0.01,
    gradient_function=calc_gradient_lasso,
    lambda_=0.5
)

print("Lasso betas:", lasso_betas)
print("Lasso iterations:", lasso_iterations)