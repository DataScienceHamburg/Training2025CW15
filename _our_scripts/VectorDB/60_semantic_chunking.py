#%% packages
from langchain_ollama import OllamaEmbeddings

from langchain_experimental.text_splitter import SemanticChunker
#%% embedding test
embeddings = OllamaEmbeddings(model='nomic-embed-text')

#%%
sample_docs = [
    "Der Hund schläft auf dem Sofa"
    "The dog sleeps on the couch",
    # please translate into french
    "le chien dort sur le canapé.",
    
]
docs_embedded = embeddings.embed_documents(sample_docs)
#%%
len(docs_embedded)

#%%
len(docs_embedded[2])

# %% Create splitter instance (4)
splitter = SemanticChunker(embeddings=embeddings)

# %% Apply semantic chunking (5)
text = "Der Hund schläft auf dem Sofa. The dog sleeps on the couch. le chien dort sur le canapé. Wir sitzen im Schulungsraum. Der Hund jagt die Katze."
    
chunks = splitter.split_text(text)
chunks

# %% check the results (6)







