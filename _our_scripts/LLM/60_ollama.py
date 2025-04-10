#%% packages
from langchain_ollama import ChatOllama

#%% Model instance
model = ChatOllama(model="gemma3:1b")

#%% Model inference
res = model.invoke("Was ist das Fraunhofer FKIE?")

#%%
res.content