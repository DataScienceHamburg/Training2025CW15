#%% packages
from datasets import load_dataset
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings

#%% data import
dataset = load_dataset("MongoDB/embedded_movies", split="train")
# %%
len(dataset)
# %% create a list of Documents
docs = []
for doc in dataset:
    title = doc['title'] if doc['title'] is not None else ""
    poster = doc['poster'] if doc['poster'] is not None else ""
    imdb_rating = doc['imdb']['rating'] if doc['imdb']['rating'] is not None else 0
    meta = {'title': title,
            'poster': poster, 
            'rating': imdb_rating}
    if doc['fullplot'] is not None:
        docs.append(Document(page_content=doc['fullplot'], metadata = meta))
    

# %% data chunking
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
# %%
len(chunks)



#%% embeddings model
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

#%% data storage
db = Chroma(persist_directory="db_movies", embedding_function=embeddings_model)

#%% add documents to db
db.add_documents(chunks)

# %% modify retriever to get five results
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5})

# %%

# %%
res = retriever.invoke("a masked avenger")
res

# %%

# %%

# %%
