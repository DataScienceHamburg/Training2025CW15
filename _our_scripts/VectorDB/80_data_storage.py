#%% packages
#%% packages
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
import os
# %%
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
parent_dir = os.path.dirname(current_dir)
text_file_path = os.path.join(parent_dir, "data", "little_women.txt")
text_file_path


# %% load a single document
loader = TextLoader(file_path=text_file_path,
                    encoding="utf-8"
                    )
docs = loader.load()
#%%
docs
# %% model dump
docs[0].model_dump()

#%% page content
docs[0].page_content

#%% 2. Recursive character text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    # separators = [""] 
)
chunks = splitter.split_documents(docs)

#%% total count
len(chunks)

#%% chunk size of all chunks
# with list comprehension
chunks_length = [len(chunk.page_content) for chunk in chunks]

#%% alternative without list comprehension (classical for-loop)
chunks_length = []
for chunk in chunks:
    chunks_length.append(len(chunk.page_content))

chunks_length

#%% visualise chunk length
import seaborn as sns
sns.histplot(chunks_length, bins=100)

#%% embeddings model
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

#%% vector store
db = Chroma(collection_name="little_women", persist_directory="db", embedding_function=embeddings_model)

#%% add all documents
db.add_documents(documents=chunks)

#%% number of docs in db
len(db.get()['ids'])

#%% use db as retriever
retriever = db.as_retriever()

# %% query db
res = retriever.invoke("Which character features has Jo?")
# %%
for r in res:
    print(r.page_content)
    print("-------------------\n")
