# docker build -t airbnb-analysis .

# use jupyter/scipy-notebook as the base image
FROM jupyter/scipy-notebook

# install extra packages
RUN conda install pytest altair_saver

# run jupyter lab instead of jupyter notebook
ENV JUPYTER_ENABLE_LAB=yes