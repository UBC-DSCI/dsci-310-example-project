#!/usr/bin/env python
# coding: utf-8

# # Predicting Airbnb nightly price from property and host data

# **Tomas Beuzen, May 2021**

# In[1]:


import pandas as pd


# ## Summary

# Here we attempt to build a regression model, using the k-nearest neighbors algorithm, which uses characteristics of an Airbnb property and host (price, bedrooms, host response rate, etc.) to predict the daily price of the property. 

# ## Introduction

# As of June, 2021, Airbnb rentals in Vancouver average \$160 a night and provide just over \$1600 in average monthly revenue to hosts {cite}`airdna_2021`. However, occupancy rates vary significantly amongst currently listed properties with some properties boasting 100% occupancy rates, while others are at 0% some months {cite}`airdna_2021`. Competitively pricing a property is a key factor influencing occupancy rates, and hence revenue {cite}`airbnb_2021`.
# 
# Here we investigate whether a machine learning algorithm can be used to predict the nightly price of an Airbnb property based on characteristics of the property and the host. Such a model could help prospective and existing hosts determine how to competitively and accurately price their new or existing property, relative to historical data, with minimal time and effort.

# ## Methods

# ### Data

# The data used to build the model contains all the active Airbnb listings for Vancouver, Canada. The data set comes from [Inside Airbnb](http://insideairbnb.com/) and is updated monthly - this project used the most recent version as of writing, from April 2021.
# 
# The data set comprises 10 features:
# 
# - The host response rate (`host_response_rate`);
# - The host acceptance rate (`host_acceptance_rate`);
# - The property location (`latitude` and `longitude`);
# - Number of bedrooms, number of beds, and number of guests the property accommodates (`bedrooms`, `beds` and `accommodates`);
# - The minimum number of nights a guests must stay (`minimum_nights`); and,
# - The review score of the property and number of reviews (`review_scores_rating`, and `number_of_reviews`).
# 
# The target variable to predict is the property's daily price (`price`). This is a number greater than 0.

# In[2]:


pd.read_csv("data/processed/airbnb_wrangled.csv").head()


# ### Splitting data into training and testing sets

# The data was split into an 80% train set and 20% test set:

# In[3]:


pd.read_csv("results/train_test_table.csv", index_col=0)


# ### Analysis

# The k-nearest neighbors algorithm (kNN) was used to build a regression model to predict the daily price of a property based on the 10 input features. As kNN is a distance-based algorithm, it was important to scale each feature to a uniform scale. As a result, each feature was normalized to be between 0 and 1 before any model fitting.
# 
# The hyperparameter `k` (number of nearest neighbors) was chosen using 10-fold cross validation with mean-absolute-error as the scoring metric. The Python programming language {cite}`python_1995` and the following Python packages were used to perform the analysis: pandas {cite}`pandas_2020`, scikit-learn {cite}`scikit_learn_2011`, altair {cite}`vanderplas_2018`, seaborn {cite}`waskom_2021`.

# ## Results and Discussion

# To look at which features might be useful to predict the price of an Airbnb property, a regression plot of each feature against the response was made (using the training data set only). From these plots, it can be seen that the features `host_response_rate` and `host_acceptance_rate` don't seem to be strongly correlated with the target, price based on the above regression plots. As a result, these were dropped from further analysis.

# ```{figure} results/regression_plots.png
# ---
# height: 600px
# name: regression-plots
# ---
# Regression plots of each feature against the target, price, from the training data.
# ```

# Values of `k` from 1 to 30 were trialled via 10-fold cross-validation to determine which value of `k` was optimal for the data. Results are shown in the figure below:

# ```{figure} results/k_optimization_plot.png
# ---
# height: 400px
# name: optimization-plot
# ---
# Results from 10-fold cross validation to choose `k`. Mean absolute error was used as the regression scoring metric.
# ```

# Results how that at values higher than `k=10` there is little change in model predictive performance in terms of mean absolute error. In the interest of parsimony (choosing a simpler model over a more complex model), a value of `k=10` was selected to train the final model:

# In[4]:


pd.read_csv("results/test_performance.csv", index_col=0)


# We see that the test performance is similar to the cross-validation performance from earlier. Our result indicates that our model has an mean absolute error of about \$41 per night which is not too bad relative to the mean and standard deviation of our training data:

# In[5]:


pd.read_csv("data/processed/airbnb_train.csv")[["price"]].describe()


# At this point, our model can provide Airbnb hosts with an estimate of how they should price their property, but it could be improved by collecting more data, or doing more intensive feature exploration and engineering.

# ## References

# ```{bibliography}
# :style: alpha
# ```
