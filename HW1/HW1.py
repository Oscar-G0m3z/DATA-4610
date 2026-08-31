import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

############ RETRIEVE DATA + LINEAR REGRESSION ##############

df = pd.read_csv('Ames Housing Data.csv')

#1 Using Linear_Regressions
df_binary = df[['SalePrice', 'Lot Area']]

# Eliminating NaN or missing input numbers
#df_binary.fillna(method ='ffill', inplace = True)

X = df_binary[['Lot Area']]
y = df_binary['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=1)

regr = LinearRegression()
regr.fit(X_train, y_train)
# print(f"R2:{regr.score(X_test, y_test)}")

################## 2 BRUTE FORCE ##############

random.seed(1)

def slr_guess_and_check(X_train, y_train, num_candidates):
    min_loss = float("inf")

    x_values = X_train['Lot Area'].to_numpy()
    y_values = y_train.to_numpy()

    for _ in range(num_candidates):

        b0 = random.uniform(0, 300000)
        b1 = random.uniform(0, 50)

        yhat = b0 + b1 * x_values

        loss = np.mean((yhat - y_values) ** 2)

        if abs(loss) < abs(min_loss):
            min_loss = loss
            min_b0 = b0
            min_b1 = b1
            num_candidates+=1

    # print("------Brute Force Method------")
    # print(f"b0: {min_b0}, b1: {min_b1}")

    return min_b0, min_b1, min_loss


# slr_guess_and_check(X_train, y_train, 10000)

############# COLLECT COMPARISON ############

# B1 / scikit-learn coefficients
correct_b0 = regr.intercept_
correct_b1 = regr.coef_[0]

# Run brute force repeatedly
results = []

for run in range(20):
    b0, b1, mse = slr_guess_and_check(
        X_train, y_train, num_candidates=10000
    )

    results.append({
        'Run': run + 1,
        'Brute Force b0': b0,
        'Brute Force b1': b1,
        'MSE': mse,
        'b0 Difference': abs(b0 - correct_b0),
        'b1 Difference': abs(b1 - correct_b1)
    })

results_df = pd.DataFrame(results)

print("Correct coefficients from LinearRegression:")
print(f"b0: {correct_b0}")
print(f"b1: {correct_b1}\n")

print(results_df)
print("\nAverage differences:")
print(results_df[['b0 Difference', 'b1 Difference']].mean())

# Get one brute-force result
brute_b0, brute_b1, brute_mse = slr_guess_and_check(
    X_train, y_train, num_candidates=10000
)

################ PLOT DIAGRAM #################

# Values to draw smooth lines across the Lot Area range
# x_line = np.linspace(
#     X_train['Lot Area'].min(),
#     X_train['Lot Area'].max(),
#     100
# )

# # Predicted SalePrice values from each model
# sklearn_line = correct_b0 + correct_b1 * x_line
# brute_line = brute_b0 + brute_b1 * x_line

# # Both fitted regression lines
# plt.plot(
#     x_line,
#     sklearn_line,
#     color='red',
#     linewidth=2,
#     label='LinearRegression (B1)'
# )

# plt.plot(
#     x_line,
#     brute_line,
#     color='blue',
#     linewidth=2,
#     label='Brute-force guess-and-check'
# )

# plt.xlabel('Lot Area')
# plt.ylabel('Sale Price')
# plt.title('Regression-Line Comparison')
# plt.legend()
# plt.show()

############### IMPROVED METHOD ###############

def slr_improvement_method(X_train, y_train, num_iterations):
    x_values = X_train['Lot Area'].to_numpy()
    y_values = y_train.to_numpy()

    b0, b1 = 0, 0
    current_loss = np.mean((b0 + b1 * x_values - y_values) ** 2)

    max_attempts = 1000

    for _ in range(num_iterations):

        # Improve b0
        attempts = 0
        b0_c = random.uniform(0, 300000)
        candidate_loss = np.mean((b0_c + b1 * x_values - y_values) ** 2)

        while candidate_loss > current_loss and attempts < max_attempts:
            b0_c = random.uniform(0, 300000)
            candidate_loss = np.mean((b0_c + b1 * x_values - y_values) ** 2)
            attempts += 1

        if candidate_loss <= current_loss:
            b0 = b0_c
            current_loss = candidate_loss

        # Improve b1
        attempts = 0
        b1_c = random.uniform(-50, 50)
        candidate_loss = np.mean((b0 + b1_c * x_values - y_values) ** 2)

        while candidate_loss > current_loss and attempts < max_attempts:
            b1_c = random.uniform(-50, 50)
            candidate_loss = np.mean((b0 + b1_c * x_values - y_values) ** 2)
            attempts += 1

        if candidate_loss <= current_loss:
            b1 = b1_c
            current_loss = candidate_loss

    return b0, b1

b0, b1 = slr_improvement_method(X_train, y_train, 100)

results_improved = []

# Run the improved method multiple times
for run in range(10):
    b0, b1 = slr_improvement_method(
        X_train,
        y_train,
        num_iterations=10
    )

    yhat = b0 + b1 * X_train['Lot Area'].to_numpy()
    mse = np.mean((yhat - y_train.to_numpy()) ** 2)

    results_improved.append({
        'Run': run + 1,
        'Improved b0': b0,
        'Improved b1': b1,
        'MSE': mse,
        'b0 Difference': abs(b0 - correct_b0),
        'b1 Difference': abs(b1 - correct_b1)
    })

improved_results_df = pd.DataFrame(results_improved)

print("Correct LinearRegression coefficients:")
print(f"b0: {correct_b0}")
print(f"b1: {correct_b1}\n")

print(improved_results_df)

print("\nAverage differences:")
print(improved_results_df[['b0 Difference', 'b1 Difference']].mean())

best_run = improved_results_df.loc[
    improved_results_df['MSE'].idxmin()
]

improved_b0 = best_run['Improved b0']
improved_b1 = best_run['Improved b1']

# x_line = np.linspace(
#     X_train['Lot Area'].min(),
#     X_train['Lot Area'].max(),
#     100
# )

# sklearn_line = correct_b0 + correct_b1 * x_line
# improved_line = improved_b0 + improved_b1 * x_line

# plt.plot(
#     x_line,
#     sklearn_line,
#     color='red',
#     label='LinearRegression (B1)'
# )

# plt.plot(
#     x_line,
#     improved_line,
#     color='blue',
#     label='Improvement method'
# )

# plt.xlabel('Lot Area')
# plt.ylabel('Sale Price')
# plt.title('Improvement Method vs. LinearRegression')
# plt.legend()
# plt.show()

data = df[['Lot Area', 'SalePrice']].dropna()

x = data['Lot Area'].to_numpy(dtype=float)
y = data['SalePrice'].to_numpy(dtype=float)

x_bar = np.mean(x)
y_bar = np.mean(y)

# Model (a): Y = b1X + b0 + error
a_b1 = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar) ** 2)
a_b0 = y_bar - a_b1 * x_bar

# Model (b): Y = bX + 5000 + error
b_beta = np.sum(x * (y - 5000)) / np.sum(x ** 2)

# Model (c): Y = bX^2 + error
c_beta = np.sum((x ** 2) * y) / np.sum(x ** 4)

print(f"Model (a): b0 = {a_b0:.4f}, b1 = {a_b1:.4f}")
print(f"Model (b): beta = {b_beta:.4f}")
print(f"Model (c): beta = {c_beta:.10f}")