# Expectation
# input_text = "which distopian novel deals with Big Brother?"

# output= {
#     "title": "1984",
#     "author": "George Orwell",
#     "publication_year": 1949,
#     "summary": "A dystopian novel set in a totalitarian society under constant surveillance, exploring themes of oppression, censorship, and individuality.",
# }

#%% packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from pydantic import BaseModel, Field
from langchain_core.output_parsers import SimpleJsonOutputParser, JsonOutputParser

#%% 
class BookOutput(BaseModel):
    title: str = Field(description="The title of the book.")
    author: str = Field(description="The author of the book.")
    year_published: int = Field(description="The year the book was published.")
    summary: str = Field(description="A brief summary of the book.")
    
#%% Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a book expert and provide key information about books."),
    ("user", "Please provide information on the book: {book}. Return the result as a JSON with the keys: title, author, year_published, summary.")
])

model = ChatGroq(model="deepseek-r1-distill-qwen-32b",
                 temperature=0.3)

parser = JsonOutputParser(pydantic_object=BookOutput)


chain = prompt_template | model | parser

#%% inference
user_prompt = "Ahab hunts a white whale"
chain.invoke({"book": user_prompt})


# %%
