#%% packages
import pymupdf4llm
# %%
# file_path = "../../FKIE_vorläufigeAgenda.pdf"
file_path = "../data/Pathfinder - Core Rulebook (5th Printing).pdf"

#%%
extracted_text = pymupdf4llm.to_markdown(doc=file_path, 
                                         page_chunks=True,
                                         force_text=True,
                                         pages=[2])
extracted_text
# %% extract images
pymupdf4llm.to_markdown(doc=file_path, pages=[0,1], write_images=True, image_path="images")
# %%
