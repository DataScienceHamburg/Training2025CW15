#%%
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
import os
# %% environment variable
os.getenv("GROQ_API_KEY")

#%% model

