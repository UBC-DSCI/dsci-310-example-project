# Milestone-1: predicting Airbnb nightly price from property and host data

Here we attempt to build a regression model to predict the nightly price of Airbnb properties using characteristics of the property and host (price, bedrooms, host response rate, etc.).

## Contents

This directory contains a Jupyter notebook and Dockerfile. All analysis (code, figures, text, etc.) is contained within the Jupyter notebook.

## Usage

Follow the steps below to reproduce the analysis.


1. Build Docker image:

    ```sh
    docker build -t airbnb-analysis .
    ```

2. Run a new container:

    ```sh
    docker run -it -p 8888:8888 -v "${PWD}":/home/jovyan/${PWD##*/} --rm airbnb-analysis
    ```

3. In the initialized Jupyter environment, find and select the `report.ipynb` file and run all cells.