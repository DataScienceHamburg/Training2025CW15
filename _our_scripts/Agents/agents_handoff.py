#%% packages
from agents import Runner, Agent
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

#%%
english_agent = Agent(
    name="english_agent",
    instructions="You are a helpful assistant that can only speak english."
)
german_agent = Agent(
    name="german_agent",
    instructions="You are a helpful assistant that can only speak german."
)

#%% triage agent
phone_operator_agent = Agent(
    name="phone_operator_agent",
    instructions="You are a helpful assistant that can handoff to the appropriate agent based on the user's language.",
    handoffs=[english_agent, german_agent]
)

#%%
response = await Runner.run(phone_operator_agent,
                            input="Ich brauche Hilfe mit meiner Bestellung.")
# %%
response.raw_responses[1].output[0].model_dump()["content"][0]["text"]
# %%
response.final_output
