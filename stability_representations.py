#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np

# Set seeds
tf.set_random_seed(42)
np.random.seed(42)
   
# Import the mLSTM babbler model
from unirep import babbler1900 as babbler
    
# Where model weights are stored.
MODEL_WEIGHT_PATH = "./data/1900_weights"


# In[2]:


batch_size = 12
model = babbler(batch_size=batch_size, model_path=MODEL_WEIGHT_PATH)


# In[ ]:


# Check that representations are reproducible
import os
import pandas as pd
from tqdm import tqdm_notebook as tqdm

# Load the saved representations
path = "./data/stability_data"
output_path = os.path.join(path, "stability_with_unirep_fusion.hdf")
existing_seqs = pd.read_hdf(output_path, key="ids").reset_index(drop=True)
existing_reps = pd.read_hdf(output_path, key="reps").reset_index(drop=True)
assert existing_seqs.shape[0] == existing_reps.shape[0]
assert np.array_equal(existing_seqs.index, existing_reps.index)

# Create reprensetations for some seqs
for index, row in tqdm(existing_seqs.iterrows(), total=existing_seqs.shape[0]):
    check_rep_1 = model.get_rep(row["sequence"])
    check_rep_1 = np.concatenate((check_rep_1[0], check_rep_1[1], check_rep_1[2]))
    check_rep_2 = model.get_rep(row["sequence"])
    check_rep_2 = np.concatenate((check_rep_2[0], check_rep_2[1], check_rep_2[2]))
    true_rep = existing_reps.iloc[index].values

    if not np.allclose(true_rep, check_rep_1, atol=0.0002) or not np.allclose(check_rep_1, check_rep_2, atol=0.0002):
        true_check_diff = abs(np.sum(true_rep - check_rep_1))
        self_run_diff = abs(np.sum(check_rep_1 - check_rep_2))
        print("{}: {} difference with saved truth".format(index, true_check_diff))
        print("{}: {} difference with self comparison".format(index, self_run_diff))


# In[3]:


import os
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

path = "./data/stability_data"
output_path = os.path.join(path, "stability_with_unirep_fusion.hdf")

existing_output = pd.DataFrame(columns=["name", "sequence", "stability"])
if os.path.isfile(output_path):
    print("Reading existing output file...")
    existing_output = pd.read_hdf(output_path, key="ids")
    print("Got {} existing data points".format(existing_output.shape[0]))
    duplicates = existing_output.duplicated(subset=["sequence"])
    assert True not in duplicates.values

new_ids_output = pd.DataFrame(columns=["name", "sequence", "stability"])
new_reps_output = pd.DataFrame(columns=list(range(0, 5700)))

for filename in os.listdir(path):
    if filename.endswith(".txt"):
        print("Processing data from {}".format(filename))
        df = pd.read_table(os.path.join(path, filename))
        model.get_rep(df["sequence"].values)
        for index, row in tqdm(df.iterrows(), total=df.shape[0]): # TODO: Parallelize this
            if index != 0 and index % 20 == 0:
                assert new_ids_output.shape[0] == new_reps_output.shape[0]
                if new_ids_output.shape[0] > 0:
                    print("Appending {} points...".format(new_ids_output.shape[0]))
                    new_ids_output.to_hdf(output_path, index=False, mode="a", key="ids", format="table", append=True)
                    new_ids_output = pd.DataFrame(columns=["name", "sequence", "stability"])
                    new_reps_output.to_hdf(output_path, index=False, mode="a", key="reps", format="table", append=True)
                    new_reps_output = pd.DataFrame(columns=list(range(0, 5700)))
            # If there is no existing data or the existing data already contains this sequence, ignore it
            if existing_output.empty or not row["sequence"] in existing_output["sequence"].values:
                if model.is_valid_seq(row["sequence"], max_len=500):
                    unirep_fusion = model.get_rep(row["sequence"])
                    unirep_fusion = np.concatenate((unirep_fusion[0], unirep_fusion[1], unirep_fusion[2]))
                    print(unirep_fusion.shape)
                    if "consensus_stability_score" in df.columns:
                        stability_score = row["consensus_stability_score"]
                    else:
                        stability_score = row["stabilityscore"]
                    new_ids_output.loc[len(new_ids_output)]=[row["name"], row["sequence"], stability_score]
                    new_reps_output.loc[len(new_reps_output)]=unirep_fusion


# In[ ]:


ids = pd.read_hdf(output_path, key="ids")
print("{} points in ids".format(ids.shape[0]))
reps = pd.read_hdf(output_path, key="reps")
print("{} points in reps".format(reps.shape[0]))
print(reps.iloc[0])

