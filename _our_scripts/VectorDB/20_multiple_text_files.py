#%% packages
from langchain.document_loaders import TextLoader, DirectoryLoader
import os
# %% get "data" folder
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
parent_dir = os.path.dirname(current_dir)
text_file_path = os.path.join(parent_dir, "data")
text_file_path

#%% 
dir_loader = DirectoryLoader(path=text_file_path,
                             glob="**/*.txt",
                             loader_cls=TextLoader, 
                             loader_kwargs={"encoding": "utf-8"
                                            })
# %%
docs = dir_loader.load()
docs