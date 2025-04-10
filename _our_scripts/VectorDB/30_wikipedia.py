# Exercise

# lade artikel zu "Artificial Intelligence"

# 10 Artikel

# mit jeweils max. 10000 Zeichen

#%% packages
from langchain.document_loaders import WikipediaLoader

article_title = "Künstliche Intelligenz"
loader = WikipediaLoader(query=article_title, 
                         lang="de",
                         load_max_docs=10,
                         doc_content_chars_max=1000)
docs = loader.load()
#%%
docs