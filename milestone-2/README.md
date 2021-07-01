# Milestone 2: predicting Airbnb nightly price from property and host data

[![Python](https://img.shields.io/badge/python-3.9-blue)]()
[![codecov](https://codecov.io/gh/TomasBeuzen/airbnb_prediction/branch/main/graph/badge.svg)](https://codecov.io/gh/TomasBeuzen/airbnb_prediction)
[![Documentation Status](https://readthedocs.org/projects/airbnb_prediction/badge/?version=latest)](https://airbnb_prediction.readthedocs.io/en/latest/?badge=latest)
[![Build](https://github.com/TomasBeuzen/airbnb_prediction/workflows/build/badge.svg)](https://github.com/TomasBeuzen/airbnb_prediction/actions/workflows/build.yml)

Here we provide a package, `airbnb_prediction`, containing functionality to build a regression model to predict the nightly price of Airbnb properties using characteristics of the property and host (price, bedrooms, host response rate, etc.).

## Contents

In this project, we have abstracted code and documentation required to run our analysis into a package `airbnb_prediction`. As a result, our analysis is self-contained, transparent about dependencies, and easy to test and distribute (if desired).

## Installation

The package can be installed using `poetry`:

    ```bash
    poetry install
    ```

### Rendering report and documentation

Functionality of `airbnb_prediction` is demonstrated in the documentation, which can be found at: `docs/_build/html/index.html`.

To build the documentation from source, run the following from the project root:

    ```bash
    make html -C docs
    ```

### Running tests

To run package tests, run the following from the project root:

    ```sh
    pytest tests/
    ```

## Dependencies

Dependencies are listed in the `pyproject.toml` file. The exact versions of all dependencies used to develop the package in its current state are listed in `poetry.lock`.

## Contributors

We welcome and recognize all contributions. You can see a list of current contributors in the [contributors tab](https://github.com/TomasBeuzen/airbnb_prediction/graphs/contributors).

### Credits

This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
