#%% packages
from agents import Runner, Agent
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))


#%% define the agents
english_agent = Agent(
    name="English Agent",
    instructions="You are a helpful agent and only speak in English.",
    # set the model to llama3.1
    model="llama3.1",
)

german_agent = Agent(
    name="German Agent",
    instructions="You are a helpful agent and only speak in German.",
)

#%% triage agent
phone_operator_agent = Agent(
    name="Phone Operator Agent",
    instructions="You are a helpful agent that can handoff to the appropriate agent based on the user's language.",
    handoffs=[english_agent, german_agent],
)

#%% define the runner
response = await Runner.run(phone_operator_agent, 
                      input="Ich brauche Hilfe mit meiner Buchung.")


# %% run the request
response.final_output

# %% check all raw responses
response.raw_responses
