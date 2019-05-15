
# coding: utf-8

# In[32]:


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


# In[36]:


test_set_protein_names = ["hYAP65", "EHEE_rd2_0005", "HHH_rd3_0138", "HHH_rd2_0134", "villin", "EEHEE_rd3_1498", "EEHEE_rd3_0037", "HEEH_rd3_0872"]

test_proteins = pd.DataFrame(columns=["sequence", "stability", "name"])
test_reps = pd.DataFrame(columns=list(range(0, 5700)))
train_proteins = pd.DataFrame(columns=["sequence", "stability", "name"])
train_reps = pd.DataFrame(columns=list(range(0, 5700)))
# Iterate the ids and put them into train or test based on matching the 
for index, row in tqdm(ids.iterrows(), total=ids.shape[0]):
    if any(protein in row["name"] for protein in test_set_protein_names):
        test_proteins.loc[len(test_proteins)]=[row["sequence"], row["stability"], row["name"]]
        test_reps.loc[len(test_reps)]=reps.iloc[index]
    else:
        train_proteins.loc[len(train_proteins)]=[row["sequence"], row["stability"], row["name"]]
        train_reps.loc[len(train_reps)]=reps.iloc[index]


# In[31]:


print(train_proteins.shape)
print(train_reps.shape)
print(test_proteins.shape)
print(test_reps.shape)
print(test_proteins.iloc[0])
print(test_reps.iloc[0])
print(ids.iloc[0])
print(reps.iloc[0])


# In[33]:


from data_utils import aa_seq_to_int

seq_ints = []

for seq in ids["sequence"]:
    seq_int = aa_seq_to_int(seq)
    seq_ints += [seq_int]
    
X_train, X_test, y_train, y_test = train_test_split(seq_ints, ids["stability"], test_size=0.15)

cv = 3
# LassoLars usage: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html#sklearn.linear_model.LassoLars
reg_sequences = linear_model.LassoLarsCV(cv=cv)
print("Training...")
reg_sequences.fit(X_train, y_train)

score_train = reg_sequences.score(X_train, y_train)
score_test = reg_sequences.score(X_test, y_test)
print("Train score: {}".format(score_train))
print("Test score: {}".format(score_test))


# In[39]:


from data_utils import aa_seq_to_int

train_seq_ints = []
test_seq_ints = []

for seq in train_proteins["sequence"]:
    seq_int = aa_seq_to_int(seq)
    train_seq_ints += [seq_int]
    
for seq in test_proteins["sequence"]:
    seq_int = aa_seq_to_int(seq)
    test_seq_ints += [seq_int]
    
print(len(train_seq_ints))
print(len(test_seq_ints))

cv = 3
# LassoLars usage: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html#sklearn.linear_model.LassoLars
reg_sequences = linear_model.LassoLarsCV(cv=cv)
print("Training...")
reg_sequences.fit(train_seq_ints, train_proteins["stability"])

score_train = reg_sequences.score(train_seq_ints, train_proteins["stability"])
score_test = reg_sequences.score(test_seq_ints, test_proteins["stability"])
print("Train score: {}".format(score_train))
print("Test score: {}".format(score_test))


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(reps, ids["stability"], test_size=0.15)
print("{} points in test set".format(X_train.shape[0]))

cv = 3
reg_uni = linear_model.LassoLarsCV(cv=cv)
print("Training {}-fold cross validated LassoLars with EclRep Fusion representations as input...".format(cv))
reg_uni.fit(X_train, y_train)


# In[1]:


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

# In[28]:


cv = 3
reg_uni = linear_model.LassoLarsCV(cv=cv)
print("Training {}-fold cross validated LassoLars with EclRep Fusion representations as input...".format(cv))
print(train_reps.shape)
reg_uni.fit(train_reps, train_proteins["stability"])


# In[29]:


score_train = reg_uni.score(train_reps, train_proteins["stability"])
score_test = reg_uni.score(test_reps, test_proteins["stability"])
print("Train score: {}".format(score_train))
print("Test score: {}".format(score_test))


# In[ ]:




