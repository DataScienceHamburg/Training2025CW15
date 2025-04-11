#%%
from autogen import ConversableAgent
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
import os

# %%
llm_config = {
    "config_list": [
        {
        "model": "cogito:3b",
        "api_key": "ollama",
        "base_url": "http://localhost:11434/v1"
        }
    ]
}
# %%
my_assistant = ConversableAgent(
    name="Igor",
    llm_config=llm_config,
    system_message="""
    You are a helpfull AI assistant. Answer concise on the user query. Use a tool for providing information on letter counts.
    """,
    human_input_mode="NEVER"
)

#%% Register the tool
def count_letters(word: str, letter: str):
    return word.lower().count(letter.lower())
# %%
my_assistant.register_for_llm(
    name='count_letters',
    description="Returns the number of a specific letter in a word. Return 'Task COMPLETED' when the task is done"
    )(count_letters)

#%%
# my_assistant.register_for_execution(
    # name="count_letters")(count_letters)
# %%

user_proxy = ConversableAgent(
    name="user_proxy",
    llm_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: msg.get("content") is not None and "Task COMPLETED" in msg["content"]
)
user_proxy.register_for_execution(
    name="count_letters")(count_letters)

# %%
result = user_proxy.initiate_chat(
    recipient=my_assistant,
    message="How many L's in 'lollapalooza'",
    # max_turns=3
)
#%%
result.chat_history
# %%
result
# %%
