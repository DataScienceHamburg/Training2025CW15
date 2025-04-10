#%%
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
import os
from langchain_groq import ChatGroq
# %% environment variable
os.getenv("GROQ_API_KEY")

#%% model
MODEL_NAME = "llama-3.3-70b-versatile"

model = ChatGroq(model=MODEL_NAME,
                 temperature=2)


# %% model inference
res = model.invoke("Was ist das Fraunhofer FKIE?")

# %%
from pprint import pprint
pprint(res.content)
# %%
res.model_dump()

#%%