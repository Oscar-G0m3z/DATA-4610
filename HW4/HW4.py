import numpy as np
import pandas as pd

data = pd.read_csv("Heart Attack Data.csv")

# Response variable
Y = data["output"].to_numpy()

X = np.column_stack([
    np.ones(len(data)),
    data["chol"].to_numpy()
])

def calc_gradient_lr(Y, X, beta_0, beta_1):
    """
    Calculates the two gradient values for cross-entropy
    logistic-regression loss.
    """

    beta = np.array([beta_0, beta_1])

    # Calculate the linear predictor:
    # z_i = beta_0 + beta_1*x_i
    z = X @ beta

    # Prevent overflow when calculating the exponential
    z = np.clip(z, -500, 500)

    # Calculate predicted probabilities using the sigmoid function
    p_hat = 1 / (1 + np.exp(-z))

    # Cross-entropy gradient:
    # gradient_j = sum((p_hat_i - Y_i)*x_ij)
    gradients = X.T @ (p_hat - Y)

    return gradients


# Verify the gradient function using beta_0 = 1 and beta_1 = 0
initial_gradient = calc_gradient_lr(
    Y=Y,
    X=X,
    beta_0=1,
    beta_1=0
)

print("Part B1")
print("Initial gradient:", initial_gradient)
print("Gradient for beta_0:", initial_gradient[0])
print("Gradient for beta_1:", initial_gradient[1])

def gradient_descent(
    Y,
    X,
    eta,
    gradient_function,
    tolerance=0.1,
    max_iterations=1000
):
    """
    Performs gradient descent for a model with an intercept
    and one predictor.
    """

    # Required initial coefficient values
    beta_0 = 1.0
    beta_1 = 0.0

    num_iterations = 0

    while num_iterations < max_iterations:

        # Calculate the current gradient
        gradients = gradient_function(
            Y,
            X,
            beta_0,
            beta_1
        )

        max_gradient = np.max(np.abs(gradients))

        if max_gradient <= tolerance:
            break

        # Gradient-descent coefficient updates
        beta_0 = beta_0 - eta * gradients[0]
        beta_1 = beta_1 - eta * gradients[1]

        num_iterations += 1

    betas = np.array([beta_0, beta_1])

    return betas, num_iterations


# Fit the logistic-regression model
betas, iterations = gradient_descent(
    Y=Y,
    X=X,
    eta=0.01,
    gradient_function=calc_gradient_lr
)


# Calculate the final gradient to determine why it stopped
final_gradient = calc_gradient_lr(
    Y=Y,
    X=X,
    beta_0=betas[0],
    beta_1=betas[1]
)

largest_final_gradient = np.max(np.abs(final_gradient))


if largest_final_gradient <= 0.1:
    stopping_condition = (
        "The largest absolute gradient component was at most 0.1."
    )
else:
    stopping_condition = (
        "The optimizer reached the 1,000-iteration safeguard."
    )


# Display the Part B2 and B3 results
print("\nPart B2 and B3")
print("Fitted beta_0:", betas[0])
print("Fitted beta_1:", betas[1])
print("Number of iterations:", iterations)
print("Final gradient:", final_gradient)
print("Largest absolute final gradient:", largest_final_gradient)
print("Stopping condition:", stopping_condition)