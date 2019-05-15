
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np

# Set seeds
random_seed = 42
tf.set_random_seed(random_seed)
np.random.seed(random_seed)
   
# Import the mLSTM babbler model
from unirep import babbler1900 as babbler
    
# Where model weights are stored.
MODEL_WEIGHT_PATH = "./data/1900_weights"


# In[2]:


batch_size = 12
model = babbler(batch_size=batch_size, model_path=MODEL_WEIGHT_PATH)


# We're testing using three seeds:
# - `MAWRQNTRYSRIEAIK` (16aas), propeptide inhibitory fragment
# - `MRYSRIEAIKIQILSKLRL` (19 aas), propeptide inhibitory fragment
# - `MDFGLDCDEHSTESRC` (12 aas), a piece of the active domain of GDF8

# In[ ]:


import pandas as pd
import itertools
from tqdm import tqdm_notebook as tqdm

# TODO: Mutationally scan these seeds?
propep_seed_1 = "MAWRQNTRYSRIEAIK"
propep_seed_2 = "MRYSRIEAIKIQILSKLRL"
active_seed = "MDFGLDCDEHSTESRC"

seeds = [propep_seed_1, propep_seed_2, active_seed]
temps = [0.01, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
lengths = [400]
params = [seeds, temps, lengths]
hyperparams = list(itertools.product(*params))
print(hyperparams)
babbles = pd.DataFrame(columns=["seed", "random_seed", "max_length", "temp", "generated_sequence"])
                                
random_seed=42

for hyperparam in tqdm(hyperparams):
    seed = hyperparam[0]
    temp = hyperparam[1]
    length = hyperparam[2]
    babble = model.get_babble(seed=seed, length=400, temp=temp)
    if "stop" in babble:
        babble = babble.split("stop", 1)[0]
        if babble not in babbles["generated_sequence"].values:
            # TODO: Get stability prediction on this babble
            # TODO: Get binding predictions on this babble
            babbles.loc[len(babbles)]=[seed, random_seed, length, temp, babble]
        else:
            print("Duplicate babble!")
    else:
        print("No stop:")
        print("seed:{}, length:{}, temp:{}".format(seed, length, temp))
        print(babble)

babbles.to_csv("./data/babbled_sequences.csv", index=False)


# In[ ]:




