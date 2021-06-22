from prefect import task, Flow

@task
download data and save locally
make plots
make model
build report



def say_hello():
    print("Hello, world!")

with Flow("Build report") as flow:
    first_result = add(1, y=2)
    second_result = add(x=first_result, y=100)