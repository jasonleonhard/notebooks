# A linear regression line y = mx + b
# can be used to find the relationship between X and Y

# import libs
from sklearn import datasets, linear_model
import numpy as np

# load data
diabetes = datasets.load_diabetes()
diabetes

# use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[:-20]
diabetes_X_train
diabetes_X_test

# split targets into training and test sets
diabetes_Y_train = diabetes.target[:-20]
diabetes_Y_test  = diabetes.target[-20:]
diabetes_Y_train
diabetes_Y_test

# linear regresson
regr = linear_model.LinearRegression()
regr

# train model using training sets
regr.fit(diabetes_X_train, diabetes_Y_train)
regr
