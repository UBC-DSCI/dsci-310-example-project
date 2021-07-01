import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error


def load_data(
    url="http://data.insideairbnb.com/canada/bc/vancouver/2021-04-12/data/listings.csv.gz",
    save=False,
    save_location="data/raw/airbnb.csv",
    verbose=True
):
    """Load data from InsideAirbnb.

    Data is downloaded from remote location if it doesn't already
    exist and can be saved if specified. If it does exist, it is
    loaded from the local source.

    Parameters
    ----------
    url : str, optional
        URL to data on InsideAirbnb.
    save : bool, optional
        Save downloaded file locally as data/airbnb.csv. By default, False.
    save_location : str, optional
        The download/save location of file.
    verbose : bool, optional
        Print progress information. By default, True.

    Returns
    -------
    pd.DataFrame
        DataFrame of Airbnb listing data.
    """
    if os.path.isfile(save_location):
        if verbose:
            print("Data already downloaded, loading from local source...")
        data = pd.read_csv(save_location)
    else:
        if verbose:
            print("Downloading data from online source...")
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
        data = pd.read_csv(url, usecols=cols, compression="gzip")
        if save:
            if verbose:
                print(f"Saving to {save_location}...")
            data.to_csv(save_location, index=False)
    return data


def wrangle_data(dataframe, save=False, save_location="data/processed/airbnb_wrangled.csv"):
    """Wrangle data into a format suitable for ML.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame of Airbnb listing data to wrangle.
    save : bool, optional
        Save wrangled data locally, by default, False.
    save_location : str, optional
        The download/save location of file.

    Returns
    -------
    pd.DataFrame
        A copy of dataframe wrangled into a format suitable for ML.
    """
    if os.path.isfile(save_location) and not save:
        print("Data already downloaded, loading from local source...")
        data_wrangled = pd.read_csv(save_location)
    else:
        data_wrangled = (
            dataframe.query("number_of_reviews >= 1")
            .dropna()
            .reset_index(drop=True)
            .copy()
        )
        data_wrangled.loc[:, "host_response_rate"] = (
            data_wrangled.loc[:, "host_response_rate"]
            .replace({"%": ""}, regex=True)
            .astype(int)
        )
        data_wrangled.loc[:, "host_acceptance_rate"] = (
            data_wrangled.loc[:, "host_acceptance_rate"]
            .replace({"%": ""}, regex=True)
            .astype(int)
        )
        data_wrangled.loc[:, "price"] = (
            data_wrangled.loc[:, "price"]
            .replace({"[$,]": ""}, regex=True)
            .astype(float)
        )
        data_wrangled = data_wrangled.query("price < 500").rename(
            columns={"review_scores_rating": "rating"}
        )
        if save:
            data_wrangled.to_csv(save_location, index=False)
    return data_wrangled


def split_data(
    dataframe,
    test_fraction=0.2,
    random_state=123,
    save=False,
    save_location="data/processed/airbnb",
):
    """Split data into train and test sets.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame of Airbnb listing data to wrangle.
    test_fraction : float
        Fraction of data to be reserved for testing, by default 0.2
    random_state : int, optional
        Random seed, by default 123
    save : bool, optional
        Whether to save train and test sets as csv files, by default False
    save_location : str, optional
        The download/save directory of train/test splits.

    Returns
    -------
    train : pd.DataFrame
        Train data set.
    test : pd.DataFrame
        Test data set.
    """
    if os.path.isfile(save_location + "_train.csv") and os.path.isfile(
        save_location + "test.csv"
    ) and not save:
        print("Data already split, reading from local source...")
        train, test = pd.read_csv(save_location + "_train.csv"), pd.read_csv(
            save_location + "_test.csv"
        )
    else:
        train, test = train_test_split(
            dataframe, test_size=test_fraction, random_state=random_state
        )
        if save:
            train.to_csv(save_location + "_train.csv", index=False)
            test.to_csv(save_location + "_test.csv", index=False)
    return train, test


def train_test_table(
    train_df, test_df, save=False, save_location="results/train_test_table.csv"
):
    """Print train and test data summary statistics.

    Parameters
    ----------
    train_df : pd.DataFrame
        Train data.
    test_df : pd.DataFrame
        Test data.
    save : bool, optional
        Whether to table as csv file, by default False
    save_location : str, optional
        The save location of file.

    Returns
    -------
    pd.DataFrame
        DataFrame of train and test data summary statistics.
    """
    test_fraction = len(test_df) / (len(train_df) + len(test_df))
    table = pd.DataFrame(
        {
            "Partition": ["Train", "Test"],
            "Fraction": [1 - test_fraction, test_fraction],
            "Median price": [train_df["price"].median(), test_df["price"].median()],
            "Mean price": [train_df["price"].mean(), test_df["price"].mean()],
            "Std price": [train_df["price"].std(), test_df["price"].std()],
        }
    ).round(1)
    if save:
        table.to_csv(save_location, index=False)
    return table


def df_to_xy(train_df, test_df, normalize=True):
    """Split dataframes into X (features) and y (target) subsets.

    Parameters
    ----------
    train_df : pd.DataFrame
        Train data.
    test_df : pd.DataFrame
        Test data.
    normalize : bool, optional
        Whether to normalize features between 0 and 1, by default True.

    Returns
    -------
    X_train : pd.DataFrame
        Train feature data.
    y_train : pd.DataFrame
        Train target data.
    X_test : pd.DataFrame
        Test feature data.
    y_test : pd.DataFrame
        Test target data.
    """
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


def k_optimization(
    X_train,
    y_train,
    k_range=range(1, 31),
    cv_folds=10,
    save=False,
    save_location="results/k_optimization.csv",
):
    """Report cross-validation results for kNN regression model.

    Parameters
    ----------
    X_train : array-like
        Feature data of shape (n_samples, n_features).
    y_train : array-like
        Target values of shape (n_samples,).
    k_range : array-like, optional
        Iterable of values of k to trial for model fitting, by default range(1, 30).
    cv_folds : int, optional
        Number of folds to use in cross-validation, by default 10.
    save : bool, optional
        Whether to save table as csv file, by default False
    save_location : str, optional
        The save location of file.

    Returns
    -------
    pd.DataFrame
        DataFrame of cross-validation results.
    """
    cv_results = []
    for k in k_range:
        cv = -cross_val_score(
            KNeighborsRegressor(n_neighbors=k),
            X_train,
            y_train,
            cv=cv_folds,
            scoring="neg_mean_absolute_error",
        )
        cv_results.append((k, cv.mean(), cv.min(), cv.max()))
    cv_results_df = pd.DataFrame(cv_results, columns=["k", "Mean", "Min", "Max"])
    if save:
        cv_results_df.to_csv(save_location, index=False)
    return cv_results_df


def test_model(X_train, y_train, X_test, y_test, k=5, save=False, save_location="results/test_performance.csv"):
    """Report cross-validation results for kNN regression model.

    Parameters
    ----------
    X_train : array-like
        Train feature data of shape (n_samples, n_features).
    y_train : array-like
        Train target values of shape (n_samples,).
    X_test : array-like
        Test feature data of shape (n_samples, n_features).
    y_test : array-like
        Test target values of shape (n_samples,).
    k : integer, optional
        Value of k to use in sklearn.neighbors.KNeighborsRegressor.
    save : bool, optional
        Whether to save result as csv file, by default False
    save_location : str, optional
        The save location of file.

    Returns
    -------
    float
        Test score.
    sklearn.neighbors.KNeighborsRegressor
        Trained model.
    """
    model = KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    if save:
        pd.DataFrame(dict(k=[k], mae=[f"${mae:.2f}"]), index=["Test data"]).to_csv(save_location, index=False)
    return mae, model
