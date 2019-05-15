
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from scipy import stats
from tqdm import tqdm_notebook as tqdm

np.random.seed(42)

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

path = "./data/stability_data"
output_path = os.path.join(path, "stability_with_unirep_fusion.hdf")

ids = pd.read_hdf(output_path, key="ids")
reps = pd.read_hdf(output_path, key="reps")

print("X: {}".format(reps.shape))
print("Y: {}".format(ids["stability"].shape))


# In[17]:


test_set_protein_names = ["hYAP65", "EHEE_rd2_0005", "HHH_rd3_0138", "HHH_rd2_0134", "villin", "EEHEE_rd3_1498", "EEHEE_rd3_0037", "HEEH_rd3_0872"]

test_indices = ids.name.str.contains('|'.join(test_set_protein_names))

test_proteins = ids[test_indices]
print(test_proteins.shape)
test_reps = reps[test_indices]
print(test_reps.shape)
train_proteins = ids[~test_indices]
print(train_proteins.shape)
train_reps = reps[~test_indices]
print(train_reps.shape)

assert test_proteins.shape[0]+train_proteins.shape[0] == ids.shape[0]


# In[51]:


from data_utils import aa_seq_to_int

seq_ints = ids["sequence"].apply(aa_seq_to_int).tolist()
    
X_train, X_test, y_train, y_test = train_test_split(seq_ints, ids["stability"], test_size=0.15)

cv = 20
# LassoLars usage: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html#sklearn.linear_model.LassoLars
reg_sequences = linear_model.LassoLarsCV(cv=cv)
print("Training...")
print(len(X_train))
print(y_train.shape)
reg_sequences.fit(X_train, y_train)

score_train = reg_sequences.score(X_train, y_train)
score_test = reg_sequences.score(X_test, y_test)
print("Train score: {}".format(score_train))
print("Test score: {}".format(score_test))


# In[59]:


from data_utils import aa_seq_to_int

train_ints = train_proteins["sequence"].apply(aa_seq_to_int).tolist()
test_ints = test_proteins["sequence"].apply(aa_seq_to_int).tolist()

cv = 20
# LassoLars usage: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html#sklearn.linear_model.LassoLars
reg_sequences = linear_model.LassoLarsCV(cv=cv)
print("Training...")
print(len(train_ints))
print(train_proteins["stability"].shape)
reg_sequences.fit(train_ints, train_proteins["stability"])

score_train = reg_sequences.score(train_ints, train_proteins["stability"])
score_test = reg_sequences.score(test_ints, test_proteins["stability"])
print("Train score: {}".format(score_train))
print("Test score: {}".format(score_test))


# In[53]:


X_train, X_test, y_train, y_test = train_test_split(reps, ids["stability"], test_size=0.15)
print("{} points in test set".format(X_train.shape[0]))

cv = 10
reg_uni = linear_model.LassoLarsCV(cv=cv)
print("Training {}-fold cross validated LassoLars with EclRep Fusion representations as input...".format(cv))
reg_uni.fit(X_train, y_train)


# In[54]:


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


# ![Capture.PNG](/notebooks/Capture.PNG)

# In[56]:


cv = 10
reg_uni = linear_model.LassoLarsCV(cv=cv)
print("Training {}-fold cross validated LassoLars with EclRep Fusion representations as input on {} input points...".format(cv, train_reps.shape))
reg_uni.fit(train_reps, train_proteins["stability"])


# In[57]:


score_train = reg_uni.score(train_reps, train_proteins["stability"])
score_test = reg_uni.score(test_reps, test_proteins["stability"])
print("Train score: {}".format(score_train))
print("Test score: {}".format(score_test))

