# docker build -t airbnb-analysis-final .

# use jupyter/scipy-notebook as the base image
FROM jupyter/scipy-notebook

# install local package
ADD *.whl .
RUN pip install *.whl

# install report build dependencies
RUN pip install prefect jupyter-book

# run jupyter lab instead of jupyter notebook
ENV JUPYTER_ENABLE_LAB=yes