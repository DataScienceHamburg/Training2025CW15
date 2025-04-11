#%% packages
from autogen import ConversableAgent
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
import os
#%% LLM config
llm_config = {
    "config_list": [
        {
            "api_type": "openai",
            "model": "gpt-4o-mini",
            "api_key": os.environ["OPENAI_API_KEY"]
        },        
        {
            "model": "llama3-8b-8192",
            "api_key": os.getenv("GROQ_API_KEY"),
            "base_url": "https://api.groq.com/openai/v1"
        }
    ]
}
# %% agent setup
jack_flat_earther = ConversableAgent(
    name="Jack",
    llm_config=llm_config,
    system_message="""
    Du glaubst fest, dass die Erde flach ist.
    Du versuchst andere von deiner Meinung zu überzeugen. Mit jeder Antwort wirst du frustrierter und unverständlicher, dass die andere Person es nicht versteht.
    """,
    human_input_mode="NEVER"
)

alice_scientist = ConversableAgent(
    name="Alice",
    llm_config=llm_config,
    system_message="""
    Du bist eine rational denkende Wissenschaftlerin und bist davon überzeugt, dass die Erde nahezu rund ist.
    Antworte freundlich, kurz und bündig.
    Du bist stichhaltigen Argumenten gegenüber offen.
    """
)
#%%
result = jack_flat_earther.initiate_chat(
    recipient=alice_scientist,
    message="Hallo, wie weit ist es von Wachtberg bis zum Rand der Erde?",
    max_turns=4
)