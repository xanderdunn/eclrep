
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

