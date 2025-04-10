#%% package
from groq import Groq
import base64
import os

#%% Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "data/find_waldo.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)
base64_image
#%% model instance
user_prompt = "Where is Waldo in the image i provided? Please give me the coordinates of Waldo in the image, e.g. top left corner is (0, 0), bottom right is (1, 1)."

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model="llama-3.2-90b-vision-preview",
)

print(chat_completion.choices[0].message.content)
# %%
