import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score


def load_data(url):
    cols = [
        "host_response_rate",
        "host_acceptance_rate",
        "latitude",
        "longitude",
        "bedrooms",
        "beds",
        "minimum_nights",
        "accommodates",
        "price",
        "review_scores_rating",
        "number_of_reviews",
    ]
    return pd.read_csv(url, usecols=cols, compression="gzip")


def wrangle_data(data):
    data_wrangled = (
        data.query("number_of_reviews >= 1").dropna().reset_index(drop=True).copy()
    )
    data_wrangled.loc[:, "host_response_rate"] = (
        data_wrangled.loc[:, "host_response_rate"]
        .replace({"\%": ""}, regex=True)
        .astype(int)
    )
    data_wrangled.loc[:, "host_acceptance_rate"] = (
        data_wrangled.loc[:, "host_acceptance_rate"]
        .replace({"\%": ""}, regex=True)
        .astype(int)
    )
    data_wrangled.loc[:, "price"] = (
        data_wrangled.loc[:, "price"].replace({"[\$,]": ""}, regex=True).astype(float)
    )
    data_wrangled = data_wrangled.query("price < 500").rename(
        columns={"review_scores_rating": "rating"}
    )
    return data_wrangled


def train_test_table(train_df, test_df):
    test_fraction = len(test_df) / (len(train_df) + len(test_df))
    table = pd.DataFrame({"Partition": ["Train", "Test"],
                          "Fraction": [1 - test_fraction, test_fraction],
                          "Median price": [train_df["price"].median(), test_df["price"].median()],
                          "Mean price": [train_df["price"].mean(), test_df["price"].mean()],
                          "Std price": [train_df["price"].std(), test_df["price"].std()]}).round(1)
    return table

def k_optimization(X_train, y_train, k_range=range(1, 30)):
    cv_results = []
    for k in k_range:
        cv = -cross_val_score(
            KNeighborsRegressor(n_neighbors=k),
            X_train,
            y_train,
            cv=5,
            scoring="neg_root_mean_squared_error",
        )
        cv_results.append((k, cv.mean(), cv.min(), cv.max()))
    return pd.DataFrame(cv_results, columns=["k", "Mean", "Min", "Max"])


def df_to_xy(train_df, test_df, normalize=True):
    if normalize:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(train_df.drop(columns="price"))
        X_test = scaler.transform(test_df.drop(columns="price"))
        y_train = train_df["price"]
        y_test = test_df["price"]
    else:
        X_train = train_df.drop(columns="price")
        X_test = test_df.drop(columns="price")
        y_train = train_df["price"]
        y_test = test_df["price"]
    return X_train, y_train, X_test, y_test
