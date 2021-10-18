# Milestone 2: predicting Airbnb nightly price from property and host data

Here we attempt to build a regression model to predict the nightly price of Airbnb properties using characteristics of the property and host (price, bedrooms, host response rate, etc.).

## Contents

In this directory, we have abstracted code from our original Jupyter notebook analysis into Python modules in a `src` folder. Our report now imports code from these modules (instead of containing code within the report itself), making it cleaner and more concise. With our code abstracted into modules, it is now also easier to test. We have therefore also created a `test` folder containing unit tests for our code.

## Usage

Follow the steps below to reproduce the analysis.


1. Build Docker image:

    ```sh
    docker build -t airbnb-analysis .
    ```

2. Run a new container from this directory using the following command:

    ```sh
    docker run -it -p 8888:8888 -v "${PWD}":/home/jovyan/${PWD##*/} --rm airbnb-analysis
    ```

3. (Optional) In the initialized Jupyter environment, open a new terminal and run tests using `pytest`:

    ```sh
    cd milestone-1
    pytest tests/
    ```

4. If tests pass successfuly, find and open the `report.ipynb` file and run all cells.
