#%% packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv(find_dotenv(usecwd=True))
# %% Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an ai assistant that translates English to other languages."),
    ("user", "Tranlate the sentence: {input} into {target_language}")
])

#%% Model instance
model = ChatGroq(model="llama-3.3-70b-versatile",
                 temperature=0.3)

#%% chain
chain = prompt_template | model | StrOutputParser()

#%% inference
res = chain.invoke({"input": "Hello, nice to meet you", "target_language": "Spanish"})
res
# %%
from pprint import pprint
pprint(res.content)

#%%
res