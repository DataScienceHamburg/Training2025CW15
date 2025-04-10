#%% packages
from google import genai
from google.genai import types

import PIL.Image
import os
image = PIL.Image.open('data/kiki.jpg')
#%%
user_prompt = "Welche Hunderasse ist auf dem Bild zu sehen? Ihr Gewicht liegt bei 14kg. Sie kommt aus Griechenland."
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
response = client.models.generate_content(
    model="gemini-2.5-pro-exp-03-25",
    contents=[user_prompt, image])

print(response.text)

#%%