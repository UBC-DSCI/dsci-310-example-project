import subprocess
import pandas as pd
from prefect import task, Flow
import airbnb_prediction as air


@task
def load_data():
    raw_data = air.analysis.load_data(save=True)
    return raw_data


@task
def wrangle_data(raw_data):
    wrangled_data = air.analysis.wrangle_data(raw_data, save=True)
    return wrangled_data


@task(nout=2)
def split_data(wrangled_data):
    train_data, test_data = air.analysis.split_data(wrangled_data, save=True)
    return train_data, test_data


@task
def plot_eda(train_data):
    _ = air.plotting.reg_subplots(train_data, save=True)


@task(nout=6)
def optimization(train_data, test_data):
    X_train, y_train, X_test, y_test = air.analysis.df_to_xy(
        train_data.drop(columns=["host_response_rate", "host_acceptance_rate"]),
        test_data.drop(columns=["host_response_rate", "host_acceptance_rate"]),
        normalize=True,
    )
    optimization_results = air.analysis.k_optimization(X_train, y_train, save=True)
    k_opt = optimization_results.sort_values(by="Mean", ignore_index=True)["k"].iloc[0]
    return optimization_results, k_opt, X_train, y_train, X_test, y_test


@task
def plot_optimization(optimization_results):
    _ = air.plotting.cv_results(optimization_results, save=True)


@task(nout=2)
def build_final_model(X_train, y_train, X_test, y_test, k):
    mae, model = air.analysis.test_model(
        X_train, y_train, X_test, y_test, k=k, save=True
    )


@task
def make_report():
    # first clean old build
    outp = subprocess.run(
        ["jb", "build", "."],
        capture_output=True,
    )
    # now build new report
    outp = subprocess.run(
        ["jb", "build", "."],
        capture_output=True,
    )


with Flow("Make report!") as flow:
    raw_data = load_data()
    wrangled_data = wrangle_data(raw_data)
    train_data, test_data = split_data(wrangled_data)
    _ = plot_eda(train_data)
    optimization_results, k_opt, X_train, y_train, X_test, y_test = optimization(
        train_data, test_data
    )
    _ = plot_optimization(optimization_results)
    mae, _ = build_final_model(X_train, y_train, X_test, y_test, k=k_opt)
    make_report()

state = flow.run()
assert state.is_successful(), "Error! Workflow unsuccessful!"
