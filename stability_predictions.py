#!/usr/bin/env python
# coding: utf-8

# In[29]:


# Import data
import os
import sys
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from scipy import stats
from tqdm import tqdm_notebook as tqdm

np.random.seed(42)
np.set_printoptions(threshold=sys.maxsize)

path = "./data/stability_data"
ids_path = os.path.join(path, "all_rds_ids.hdf")
reps_path = os.path.join(path, "all_rds_reps.hdf")

# Get the data
output_path = os.path.join(path, "stability_with_unirep_fusion.hdf")
ids = pd.read_hdf(output_path, key="ids").reset_index(drop=True)
reps = pd.read_hdf(output_path, key="reps").reset_index(drop=True)

print("X: {}".format(reps.shape))
print("Y: {}".format(ids["stability"].shape))


# In[30]:


# Train and test model on original sequences
from data_utils import aa_seq_to_int
import data_utils
import importlib
importlib.reload(data_utils)

seq_ints = ids["sequence"].apply(aa_seq_to_int).tolist()

X_train, X_test, y_train, y_test = train_test_split(seq_ints, ids["stability"], test_size=0.15)
print("The test set has {} data points".format(len(X_test)))

cv = 10
# LassoLars usage: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html#sklearn.linear_model.LassoLars
reg_sequences = linear_model.LassoLarsCV(cv=cv)
print("Training...")
reg_sequences.fit(X_train, y_train)

score_train = reg_sequences.score(X_train, y_train)
score_test = reg_sequences.score(X_test, y_test)
print("Train score: {}".format(score_train))
print("Test score: {}".format(score_test))


# In[31]:


# Train and test model on representations
X_train, X_test, y_train, y_test = train_test_split(reps, ids["stability"], test_size=0.15)
print("{} points in test set".format(X_test.shape[0]))

cv = 3
reg_uni = linear_model.LassoLarsCV(cv=cv)
print("Training {}-fold cross validated LassoLars with EclRep Fusion representations as input...".format(cv))
reg_uni.fit(X_train, y_train)


# In[32]:


score_train = reg_uni.score(X_train, y_train)
score_test = reg_uni.score(X_test, y_test)
print("Train score: {}".format(score_train))
print("Test score: {}".format(score_test))

# Get Spearman's p correlation on test set predictions
test_predictions = reg_uni.predict(X_test)
spearman_test = stats.spearmanr(test_predictions, y_test)
print("Spearman's p on test set: {}".format(spearman_test))

# Plot the predictions vs. measured values
import plotly.plotly as py
import plotly
import plotly.graph_objs as go

plotly.tools.set_credentials_file(username='xanderdunn', api_key='GtTpDQavToMaADqeMMu4')

trace = go.Scatter(
    x = test_predictions,
    y = y_test,
    mode = 'markers'
)

py.iplot([trace], filename="Peptide Stability Prediction vs. Measured Stability")


# ![Capture.PNG](/tf/notebooks/Capture.PNG)
