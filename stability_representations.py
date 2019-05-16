#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Initialize
import tensorflow as tf
import numpy as np

# Set seeds
tf.set_random_seed(42)
np.random.seed(42)
    
# Where model weights are stored.
MODEL_WEIGHT_PATH = "./data/1900_weights"


# In[39]:


import os
from unirep import mLSTMCell1900, tf_get_shape, aa_seq_to_int
import pandas as pd


def is_valid_seq(seq, max_len=500):
    """
    True if seq is valid for the babbler, False otherwise.
    """
    l = len(seq)
    valid_aas = "MRHKDESTNQCUGPAVIFYWLO"
    if (l < max_len) and set(seq) <= set(valid_aas):
        return True
    else:
        return False

    
class babbler1900():

    def __init__(self, model_path="./data/1900_weights", batch_size=500):
        self._model_path = model_path
        self._batch_size = batch_size
        
        self._rnn = mLSTMCell1900(1900,
                    model_path=self._model_path,
                        wn=True)
        zero_state = self._rnn.zero_state(self._batch_size, tf.float32)

        self._embed_matrix = tf.get_variable(
            "embed_matrix", dtype=tf.float32, initializer=np.load(os.path.join(self._model_path, "embed_matrix:0.npy"))
        )
        
        with tf.Session() as sess:
            self._zero_state = sess.run(zero_state)
        
    def get_reps(self, seqs):
        seq_ints = [aa_seq_to_int(seq.strip())[:-1] for seq in seqs]
        lengths = [len(x) for x in seq_ints]
        tf_tensor = tf.convert_to_tensor(seq_ints)
        dataset = tf.data.Dataset.from_tensor_slices(tf_tensor).batch(self._batch_size)
        iterator = dataset.make_one_shot_iterator()
        input_tensor = iterator.get_next()

        embed_cell = tf.nn.embedding_lookup(self._embed_matrix, input_tensor)
        _output, _final_state = tf.nn.dynamic_rnn(
            self._rnn,
            embed_cell,
            initial_state=self._zero_state,
            swap_memory=True,
            parallel_iterations=1
        )
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            final_state_, hs = sess.run([_final_state, _output])
            assert final_state_[0].shape[0] == self._batch_size

            final_cell, final_hidden = final_state_
            avg_hidden = np.array([np.mean(x, axis=0) for x in hs])
            together = np.concatenate((avg_hidden, final_hidden, final_cell), axis=1)
            return together

# Given a pandas dataframe with a column "sequence", return a list of pandas dataframes grouped by sequence length
def create_batches(seqs):
    # Get the unique lengths of all these sequences
    batches = []
    lengths = seqs["sequence"].apply(lambda x: len(x))
    unique_lengths = lengths.unique()
    for unique_length in unique_lengths:
        boolean_mask = lengths == unique_length
        seqs_of_length = seqs[boolean_mask]
        batches += [seqs_of_length]
    print("There are {} batches".format(len(batches)))
    return batches
    
# Get representations for a numpy array of sequences where all sequences are the same length
def inference_on_seqs_array(seqs):
        tf.reset_default_graph()
        model = babbler1900(batch_size=seqs.shape[0])
        result = model.get_reps(seqs)
        return result

# Get representations for a pandas dataframe of sequences in coulmn "sequence"
def inference_on_seqs(seqs):
    # Check that all these sequences are valid
    valid_func = lambda x: is_valid_seq(x)
    valid_np = np.vectorize(valid_func)
    valids = valid_np(seqs["sequence"].values)
    assert False not in valids
    
    batches = create_batches(seqs)
    
    ids = None
    reps = pd.DataFrame(columns=list(range(0, 5700)))
    
    for batch in batches:
        print("Getting EclRep representations for {} sequences...".format(batch.shape[0]))
        reps_new = inference_on_seqs_array(batch["sequence"])
        print("Done")
        reps = reps.append(pd.DataFrame(reps_new))
        if ids is not None:
            ids = ids.append(batch)
        else:
            ids = batch
    return ids, reps
    
    
path = "./data/stability_data"
df = pd.read_table(os.path.join(path, "ssm2_stability_scores.txt"))
ids, results = inference_on_seqs(df)
print("Got {} results".format(results.shape))
assert results.shape[0] == seqs.shape[0]
assert results.shape[1] == 5700

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
print("Checking that these results match the saved truth...")
for index, row in tqdm(existing_seqs.iterrows(), total=existing_seqs.shape[0]):
    check_rep = results.iloc[index].values
    true_rep = existing_reps.iloc[index].values

    if not np.allclose(true_rep, check_rep, atol=0.0001):
        true_check_diff = abs(np.sum(true_rep - check_rep))
        print("{}: {} difference with saved truth".format(index, true_check_diff))


# In[19]:


import os
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

path = "./data/stability_data"
output_path = os.path.join(path, "all_stability_with_eclrep_fusion.hdf")

reps_output = pd.DataFrame(columns=list(range(0, 5700)))

# TODO: Assert that there are no duplicate sequences
# TODO: How do I match the reps with sequences?

for filename in os.listdir(path):
    if filename.endswith(".txt"):
        print("Processing data from {}".format(filename))
        df = pd.read_table(os.path.join(path, filename))
        if "consensus_stability_score" in df.columns:
            stability_name = "consensus_stability_score"
        else:
            stability_name = "stabilityscore"
        df = df[["name", "sequence", stability_name]]
        df.rename(columns={'consensus_stability_score': 'stability', 'stabilityscore': 'stability'}, inplace=True)
        results = inference_on_seqs(df["sequence"].values)
        print(results.shape)
        
#     if model.is_valid_seq(row["sequence"], max_len=500):
#         unirep_fusion = model.get_rep(row["sequence"])
#         unirep_fusion = np.concatenate((unirep_fusion[0], unirep_fusion[1], unirep_fusion[2]))
#         print(unirep_fusion.shape)


ids_output.to_hdf(output_path, index=False, mode="a", key="ids", format="fixed")
reps_output.to_hdf(output_path, index=False, mode="a", key="reps", format="fixed")


# In[ ]:


ids = pd.read_hdf(output_path, key="ids")
print("{} points in ids".format(ids.shape[0]))
reps = pd.read_hdf(output_path, key="reps")
print("{} points in reps".format(reps.shape[0]))
print(reps.iloc[0])

