#%%
from pprint import pprint
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
import os
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
# %% environment variable
os.getenv("GROQ_API_KEY")

#%% model
MODEL_NAME = "gemini-2.5-pro-exp-03-25"

model = ChatGoogleGenerativeAI(model=MODEL_NAME,
                 temperature=1)


# %% model inference
res = model.invoke("Was ist das Fraunhofer FKIE?")

# %%
pprint(res.content)
# %%
res.model_dump()

#%%