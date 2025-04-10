#%% packages
from bs4 import BeautifulSoup
import requests

#%% 
url = "https://enargus.de/"

#%%
response = requests.get(url)

#%%
response.text

#%% write to html
with open("enargus.html", "w", encoding="utf-8") as f:
    f.write(response.text)

#%%



