#%% packages
from langchain.document_loaders import TextLoader
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