# docker build -t airbnb-analysis .

# use jupyter/scipy-notebook as the base image
FROM jupyter/scipy-notebook

# install extra packages
RUN conda install pytest altair_saver jupyter-book prefect
# RUN pip install airbnb_prediction prefect jupyter-book

# run jupyter lab instead of jupyter notebook
ENV JUPYTER_ENABLE_LAB=yes