#%% packages
from langchain_ollama import OllamaEmbeddings
import numpy as np
#%% embedding test
embeddings = OllamaEmbeddings(model='mxbai-embed-large')

#%%
sample_docs = [
    "Der Hund schläft auf dem Sofa.",
    "The dog sleeps on the couch.",
    "Le chien dort sur le canapé.",
    "Quantum mechanics challenges our understanding of the universe.",
    "The cat lounged lazily on the warm windowsill.",
    "A feline relaxed comfortably on the sun soaked ledge."
]
docs_embedded = embeddings.embed_documents(sample_docs)
#%%
docs_np = np.array(docs_embedded)
docs_np.shape

#%%
docs_correlations = np.corrcoef(docs_np)
import seaborn as sns
sns.heatmap(docs_correlations, 
            annot=True,
            fmt=".1f",
            xticklabels=sample_docs,
            yticklabels=sample_docs,
            
            )
#%%