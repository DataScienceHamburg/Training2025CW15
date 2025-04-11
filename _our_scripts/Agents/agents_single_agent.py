#%% packages
from agents import Runner, Agent
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

#%%
agent = Agent(
    name="my_first_agent",
    instructions="You are a helpful assistant that can answer questions."
)

#%%
response = await Runner.run(agent,
                            input="Hello, what is Openai agents?")
# %%
response.final_output