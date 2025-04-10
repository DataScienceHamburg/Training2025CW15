#%% packages
import streamlit as st
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
user_prompt = "batman"
db = Chroma(persist_directory="db_movies", embedding_function=embeddings_model)

db.get()

#%%
retriever = db.as_retriever(
    search_kwargs={"k": 5})

# query db
res = retriever.invoke(input=user_prompt)
for r in res:
    # print(r.metadata['title'])
    print(r.metadata['title'])

#%%
st.title("Movie App")


# integrate browser in VS Code via STRG + SHIFT + P --> "Browser"
user_prompt = st.chat_input(placeholder="bitte beschreibe den Film")

if user_prompt:
    db = Chroma(persist_directory="db_movies", embedding_function=embeddings_model)
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5})
    
    # query db
    res = retriever.invoke(input=user_prompt)
    for r in res:
        # print(r.metadata['title'])
        st.text(r.metadata['title'])