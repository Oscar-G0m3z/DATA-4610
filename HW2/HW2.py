import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

df = pd.read_csv(
    "/Users/oscargomeza./Downloads/Data_4610/HW1/Ames Housing Data.csv"
)

lambda_values = [0, 1, 10, 100, 1000]

def ridge_estimator(x, y, lam):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x ** 2)
    sum_xy = np.sum(x * y)

    # Found by plugging the beta0 and beta1 equations into each other
    denominator = (n + lam) * (sum_x2 + lam) - sum_x ** 2

    beta0 = ((sum_x2 + lam) * sum_y - sum_x * sum_xy) / denominator
    beta1 = ((n + lam) * sum_xy - sum_x * sum_y) / denominator

    return beta0, beta1


def try_many_lambdas(data, x, y, lambda_vector):
    output = {}

    # Get the actual column values from the data frame
    x_values = data[x].to_numpy()
    y_values = data[y].to_numpy()

    for lam in lambda_vector:
        output[lam] = ridge_estimator(x_values, y_values, lam)

    return output

################# TESTING EACH LAMBDA ####################
print("------------ tune lambda classic ------------")
def tune_lambda_classic(df, x, y, lambdas, metric):
    # Remove rows with missing values in either selected column
    data = df[[x, y]].dropna()

    lambda_list = try_many_lambdas(data, x, y, lambdas)

    x_values = data[x].to_numpy()
    y_values = data[y].to_numpy()

    n = len(y_values)
    p = 1
    results = []

    for lam in lambda_list:
        b0, b1 = lambda_list[lam]

        # Ridge predictions
        yhat = b0 + b1 * x_values

        # Residual sum of squares
        rss = np.sum((y_values - yhat) ** 2)

        match metric:
            case 1:  # Adjusted R-squared
                tss = np.sum((y_values - np.mean(y_values)) ** 2)
                r2 = 1 - (rss / tss)
                metric_value = 1 - (1 - r2) * (n - 1) / (n - p - 1)

            case 2:  # AIC
                metric_value = n * np.log(rss / n) + 2 * (p + 1)

            case 3:  # BIC
                metric_value = n * np.log(rss / n) + (p + 1) * np.log(n)

            case _:
                raise ValueError(
                    "metric must be adjusted_r2=1, aic=2, or bic=3"
                )

        results.append({
            "lambda": lam,
            "beta0": b0,
            "beta1": b1,
            "metric": metric_value
        })

    return pd.DataFrame(results)

classic_results = tune_lambda_classic(
    df,
    "Lot Area",
    "SalePrice",
    lambda_values,
    1
)

print(classic_results)

best_model = classic_results.loc[classic_results["metric"].idxmax()]

print("Best lambda:", best_model["lambda"])
print("Beta0:", best_model["beta0"])
print("Beta1:", best_model["beta1"])
print("Adjusted R-squared:", best_model["metric"])


################### TRAINING/TEST SPLIT ####################
print("------------ tune lambda split ------------")
def tune_lambda_split(train_df, test_df, x, y, lambdas, metric):
    # Remove missing values separately from training and test data
    train_data = train_df[[x, y]].dropna()
    test_data = test_df[[x, y]].dropna()

    # Fit every lambda using TRAINING data only
    lambda_list = try_many_lambdas(train_data, x, y, lambdas)

    x_test = test_data[x].to_numpy()
    y_test = test_data[y].to_numpy()

    results = []

    for lam in lambda_list:
        b0, b1 = lambda_list[lam]

        # Predict test-data SalePrice values
        yhat = b0 + b1 * x_test

        match metric:
            case 1:  # R-squared
                rss = np.sum((y_test - yhat) ** 2)
                tss = np.sum((y_test - np.mean(y_test)) ** 2)
                metric_value = 1 - (rss / tss)

            case 2:  # Mean squared error
                metric_value = np.mean((y_test - yhat) ** 2)

            case 3:  # Mean absolute error
                metric_value = np.mean(np.abs(y_test - yhat))

            case _:
                raise ValueError(
                    "metric must be r_squared=1, mse=2, or mae=3"
                )

        results.append({
            "lambda": lam,
            "beta0": b0,
            "beta1": b1,
            "metric": metric_value
        })

    return pd.DataFrame(results)

ames_model = df[["Lot Area", "SalePrice"]].dropna()

# Random 80/20 split; random_state makes your result repeatable
train_df = ames_model.sample(frac=0.80, random_state=42)
test_df = ames_model.drop(train_df.index)

lambda_values = [0, 1, 10, 100, 1000]

split_results = tune_lambda_split(
    train_df,
    test_df,
    "Lot Area",
    "SalePrice",
    lambda_values,
    1
)

print(split_results)

best_model = split_results.loc[split_results["metric"].idxmax()]

print("Best lambda:", best_model["lambda"])
print("Beta0:", best_model["beta0"])
print("Beta1:", best_model["beta1"])
print("Test R-squared:", best_model["metric"])

############### CROSS VALIDATION ################
print("------------ tune lambda cv ------------")
def tune_lambda_cv(data, x, y, lambdas, metric, v):
    data = data[[x, y]].dropna().reset_index(drop=True)

    kf = KFold(
        n_splits=v,
        shuffle=True,
        random_state=42
    )

    all_fold_results = []

    for fold_num, (train_indexes, test_indexes) in enumerate(kf.split(data)):
        train_df = data.iloc[train_indexes]
        test_df = data.iloc[test_indexes]

        fold_results = tune_lambda_split(
            train_df,
            test_df,
            x,
            y,
            lambdas,
            metric
        )

        fold_results["fold"] = fold_num + 1
        all_fold_results.append(fold_results)

    all_fold_results = pd.concat(all_fold_results, ignore_index=True)

    # Mean metric for each lambda across all folds
    return (
        all_fold_results
        .groupby("lambda", as_index=False)["metric"]
        .mean()
    )

# 5-fold cross-validation using R-squared
cv_results = tune_lambda_cv(
    ames_model,
    "Lot Area",
    "SalePrice",
    lambda_values,
    1,      # 1 = R-squared
    5       # five folds
)

print(cv_results)

best_cv = cv_results.loc[cv_results["metric"].idxmax()]

best_lambda_cv = best_cv["lambda"]

# Refit the selected lambda using all AMES data
best_b0, best_b1 = ridge_estimator(
    ames_model["Lot Area"],
    ames_model["SalePrice"],
    best_lambda_cv
)

print("\nBest lambda:", best_lambda_cv)
print("Beta0:", best_b0)
print("Beta1:", best_b1)
print("Average 5-fold R-squared:", best_cv["metric"])

############### LOO ################
print("------------ tune lambda loo ------------")

def tune_lambda_loo(data, x, y, lambdas, metric):
    data = data[[x, y]].dropna().reset_index(drop=True)

    x_values = data[x].to_numpy()
    y_values = data[y].to_numpy()

    results = []

    # Try one lambda at a time
    for lam in lambdas:
        predictions = []

        # Leave out one row at a time
        for i in range(len(data)):
            # Row i becomes the one-row test set
            test_row = data.iloc[i]

            # Every other row becomes training data
            train_df = data.drop(index=i)

            # Fit ridge regression without row i
            b0, b1 = ridge_estimator(
                train_df[x],
                train_df[y],
                lam
            )

            # Predict the held-out row's SalePrice
            yhat_i = b0 + b1 * test_row[x]

            # Save that prediction
            predictions.append(yhat_i)

        predictions = np.array(predictions)

        # After predicting every row once, calculate one overall metric
        match metric:
            case 1:  # R-squared
                rss = np.sum((y_values - predictions) ** 2)
                tss = np.sum((y_values - np.mean(y_values)) ** 2)
                metric_value = 1 - (rss / tss)

            case 2:  # MSE
                metric_value = np.mean((y_values - predictions) ** 2)

            case 3:  # MAE
                metric_value = np.mean(np.abs(y_values - predictions))

            case _:
                raise ValueError(
                    "metric must be r_squared=1, mse=2, or mae=3"
                )

        results.append({
            "lambda": lam,
            "metric": metric_value
        })

    return pd.DataFrame(results)

loo_results = tune_lambda_loo(
    ames_model,
    "Lot Area",
    "SalePrice",
    lambda_values,
    1
)

print(loo_results)

best_loo = loo_results.loc[loo_results["metric"].idxmax()]

best_lambda_loo = best_loo["lambda"]

best_b0, best_b1 = ridge_estimator(
    ames_model["Lot Area"],
    ames_model["SalePrice"],
    best_lambda_loo
)

print("Best LOO lambda:", best_lambda_loo)
print("Beta0:", best_b0)
print("Beta1:", best_b1)
print("LOO R-squared:", best_loo["metric"])