#%% package
from groq import Groq
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
#%% model instance
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

#%% model inference
image_url = "https://upload.wikimedia.org/wikipedia/commons/f/f2/LPU-v1-die.jpg"
user_prompt = "What's in this image?"

completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    }
                }
            ]
        }
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    n=1, 
    stream=False,
    stop=None,
)

print(completion.choices[0].message)

# %%
completion.choices[0].message.content