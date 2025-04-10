#%% packages
from datasets import load_dataset
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings

#%% data import
dataset = load_dataset("MongoDB/embedded_movies", split="train")
# %%
len(dataset)
# %% create a list of Documents
docs = []
for doc in dataset:
    title = doc['title'] if doc['title'] is not None else ""
    poster = doc['poster'] if doc['poster'] is not None else ""
    imdb_rating = doc['imdb']['rating'] if doc['imdb']['rating'] is not None else 0
    meta = {'title': title,
            'poster': poster, 
            'rating': imdb_rating}
    if doc['fullplot'] is not None:
        docs.append(Document(page_content=doc['fullplot'], metadata = meta))
    

# %% data chunking
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
# %%
len(chunks)



#%% embeddings model
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

#%% data storage
db = Chroma(persist_directory="db_movies", embedding_function=embeddings_model)

#%% add documents to db
db.add_documents(chunks)

# %% modify retriever to get five results
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5})

# %%

# %% 1. Retrieval
res = retriever.invoke("a masked avenger")
res

# %% 2. Augmentation
context_info = [f"Titel: {doc.metadata['title']}; Beschreibung: {doc.page_content}" for doc in res]
context_info_str = "; ".join(context_info)
context_info_str

# %% 3. Generation
from langchain.prompts import ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a movie expert and answer about movies. You will be shown a user question, and relevant information. Answer the question using only this information. Say 'I dont know' if you do not know the answer."),
    ("user", "Question: {query}, Information: {context_info}")
])

# %% chain
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
model = ChatGroq(model="llama3-8b-8192", temperature=0)
chain = prompt_template | model | StrOutputParser()

# %%
chain.invoke({"query": "A masked avenger", "context_info": context_info_str})

# %%
