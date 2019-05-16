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


# In[11]:


import os
from unirep import mLSTMCell1900, tf_get_shape, aa_seq_to_int
import pandas as pd

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
        # Get the input sequences
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

# Create nets and run inference on a given list of sequences
def inference_on_seqs(seqs):
    len_func = lambda x: len(x)
    len_np = np.vectorize(len_func)
    seq_lengths = len_np(seqs)
    unique_lengths = np.unique(seq_lengths)
    results = None
    for unique_length in unique_lengths:
        boolean_mask = seq_lengths == unique_length
        seqs_of_length = seqs[boolean_mask]
        print("Getting EclRep representations for {} sequences of length {}...".format(seqs_of_length.shape[0], unique_length))
        tf.reset_default_graph()
        model = babbler1900(batch_size=seqs_of_length.shape[0])
        result = model.get_reps(seqs_of_length)
        print("Done")
        if results is not None:
            results = np.concatenate((results, result))
        else:
            results = result
    return results
    
    
path = "./data/stability_data"
df = pd.read_table(os.path.join(path, "ssm2_stability_scores.txt"))
seqs = df["sequence"].values
results = inference_on_seqs(seqs)
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
    check_rep = results[index]
    true_rep = existing_reps.iloc[index].values

    if not np.allclose(true_rep, check_rep, atol=0.001):
        true_check_diff = abs(np.sum(true_rep - check_rep))
        print("{}: {} difference with saved truth".format(index, true_check_diff))


# In[20]:


# Use tf.data and tf.Variable to feed data into the inference step
import os
import tensorflow as tf
from unirep import aa_seq_to_int, initialize_uninitialized
import pandas as pd

#using a placeholder
tf.set_random_seed(42)
np.random.seed(42)

# data = np.random.sample((100,2))
# tensor = tf.convert_to_tensor(data)
# dataset = tf.data.Dataset.from_tensor_slices(tensor)
# iter = dataset.make_one_shot_iterator()
# el = iter.get_next()
# with tf.Session() as sess:
#     result = sess.run(el) # output [0.37454012 0.95071431]
#     print(result)

# tf.reset_default_graph() # Reset the graph

def get_reps(model, seqs):

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
#         final_cell, final_hidden, hs = sess.run([final_cell_ts, final_hidden_ts, hs_ts])
        final_state_, hs = sess.run([model._inference_final_state, model._inference_output],
                                    feed_dict={
                                        model._minibatch_x_placeholder: [seq_ints[0]]
                                    }
                                   )

        final_cell, final_hidden = final_state_
        # Drop the batch dimension so it is just seq len by representation size
        final_cell = final_cell[0]
        final_hidden = final_hidden[0]
        hs = hs[0]
        avg_hidden = np.mean(hs, axis=0)
        return avg_hidden, final_hidden, final_cell
    
path = "./data/stability_data"
df = pd.read_table(os.path.join(path, "ssm2_stability_scores.txt"))
seqs = df["sequence"].iloc[:100].values
get_reps(model, seqs) # TODO: This is only the first 100 rows


# In[ ]:


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

