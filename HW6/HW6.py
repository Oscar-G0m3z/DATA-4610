import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


df = pd.read_csv("Heart Attack Data HW5.csv")

before = df.copy()
df["output"] = df["output"].replace({0: -1, 1: 1})

# print(pd.DataFrame({
#     "Before": before,
#     "After": df["output"]
# }).head())

######################
########## B2 ########
######################

# Input: feature matrix, {-1, 1} the response vector, 
# the current vector of coefficients, and Lambda.
def calc_gradient_svc(X, y, beta, lam):
    n = len(y)

    # calculates the subgradient of the regularized hinge-loss objective
    scores = X @ beta
    margins = y * scores

    active = margins < 1
    gradient = -(X[active].T @ y[active]) / n

    # don't penalize intercept if included
    gradient[1:] += lam * beta[1:]    
    return gradient

print(pd.DataFrame({
    "Before": before["output"],
    "After": df["output"]
}).head())

######################
########## B3 ########
######################

X = df.drop(columns="output").to_numpy(dtype=float)
y = df["output"].to_numpy(dtype=float)

# Add the intercept column
X = np.column_stack([np.ones(len(X)), X])

beta = np.ones(X.shape[1])

gradient = calc_gradient_svc(X, y, beta, lam=0.01)
print(gradient)
print("Number of gradient entries:", gradient.size)

# The returned gradient vector has one entry for each
# feature coefficient, plus one for the intercept. Each 
# entry describes how the objective changes locally when 
# that coefficient increases while the others remain fixed. 
# A positive entry means gradient descent decreases that coefficient, 
# while a negative entry means it increases that coefficient

######################
########## B4 ########
######################

def svc_objective(X, y, beta, lam):
    margins = y * (X @ beta)

    hinge_loss = np.maximum(0, 1 - margins).mean()

    # Exclude the intercept from the penalty
    penalty = (lam / 2) * np.sum(beta[1:] ** 2)

    return hinge_loss + penalty

def gradient_descent(X, y, lam, learning_rate=0.2,
                     iterations=20000):
    beta = np.ones(X.shape[1])

    objective_history = [svc_objective(X, y, beta, lam)]
    best_beta = beta.copy()
    best_loss = objective_history[0]

    for i in range(iterations):
        gradient = calc_gradient_svc(X, y, beta, lam)

        step = learning_rate / (1 + i / 100) ** 0.75
        beta = beta - step * gradient

        loss = svc_objective(X, y, beta, lam)
        objective_history.append(loss)

        if loss < best_loss:
            best_loss = loss
            best_beta = beta.copy()

    return best_beta, np.array(objective_history)

######################
########## B5 ########
######################

# Original rows 1, 3, 5, ... are training rows
X_train = X[0::2]
y_train = y[0::2]

# Original rows 2, 4, 6, ... are test rows
X_test = X[1::2]
y_test = y[1::2]

# Choose lambda using only the training data
lambda_values = [0.001, 0.01, 0.1, 1]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

best_lambda = None
best_accuracy = -1

for lam in lambda_values:
    fold_accuracies = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        beta, _ = gradient_descent(
            X_train[train_idx],
            y_train[train_idx],
            lam=lam
        )

        scores = X_train[val_idx] @ beta
        predictions = np.where(scores >= 0, 1, -1)

        accuracy = accuracy_score(y_train[val_idx], predictions)
        fold_accuracies.append(accuracy)

    mean_accuracy = np.mean(fold_accuracies)
    print(f"Lambda: {lam}, CV accuracy: {mean_accuracy:.4f}")

    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_lambda = lam

print("Selected lambda:", best_lambda)

# Fit the final model using ALL training rows
beta_final, objective_history = gradient_descent(
    X_train, y_train, lam=best_lambda
)

# Predict the test rows
test_scores = X_test @ beta_final
y_pred = np.where(test_scores >= 0, 1, -1)

print("Test predictions:", y_pred)

######################
########## B6 ########
######################

import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

print("\nTest Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(
    y_test, y_pred, pos_label=1, zero_division=0
))
print("Recall:", recall_score(
    y_test, y_pred, pos_label=1, zero_division=0
))
print("F1 score:", f1_score(
    y_test, y_pred, pos_label=1, zero_division=0
))

# ROC-AUC uses continuous scores, not predicted labels
print("ROC-AUC:", roc_auc_score(y_test, test_scores))

plt.plot(objective_history)
plt.xlabel("Iteration")
plt.ylabel("SVC objective value")
plt.title("SVC Objective vs. Iteration")
plt.grid(alpha=0.3)
plt.show()

print("Starting objective:", objective_history[0])
print("Lowest objective:", objective_history.min())
print("Last objective:", objective_history[-1])

######################
########## B7 ########
######################

train_margins = y_train * (X_train @ beta_final)
influential = train_margins <= 1

# Original one-based row numbers
train_row_numbers = np.arange(1, len(df) + 1)[0::2]

print("Training rows with signed margin <= 1:")
print(train_row_numbers[influential])
print("Number of observations:", influential.sum())