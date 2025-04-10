#%% packages
#%% packages
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
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
#%% data chunking
# 1. fixed size chunking
splitter = CharacterTextSplitter(chunk_size = 256, 
                                 chunk_overlap = 50, 
                                 separator=" ")
chunks = splitter.split_documents(docs)

#%% total count of chunks
len(chunks)

#%% 
from pprint import pprint
pprint(chunks[10].page_content)

# %%
pprint(chunks[11].page_content)

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



    











