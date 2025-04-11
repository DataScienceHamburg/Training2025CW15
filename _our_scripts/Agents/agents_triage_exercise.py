#%% packages
from agents import Agent, Runner
import asyncio
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
#%%
story_idea_generator = Agent(name="Story Idea Generator", instructions="You are a creative writer who generates original and interesting story ideas.")

character_development = Agent(name="Character Development", instructions="You take a story idea and develop detailed characters for it, including their motivations and backstories." )

plot_weaver = Agent(name="Plot Weaver", instructions="You create a detailed plot outline based on a story idea and character descriptions, focusing on engaging conflicts and resolutions.")

creative_writer = Agent(name="Creative Writer", instructions="You write a short, compelling story based on a plot outline and character descriptions.")

async def run_creative_writing():
    # Triage Agent
    triage_agent = Agent(
        name="Creative Writing Triage",
        instructions="You determine the type of creative writing task and route it to the appropriate creative writing agent.",
        handoffs=[
            story_idea_generator,
            character_development,
            plot_weaver,
            creative_writer,
        ],
    )

    #%% Example prompts that trigger different handoffs
    prompts = [
        "Generate a story idea.",
        "Develop characters for a fantasy setting.",
        "Create a plot outline for a sci-fi thriller.",
        "Write a short story about a robot who learns to love.",
    ]

    for prompt in prompts:
        response = await Runner.run(triage_agent, input=prompt)
        print(f"\n--- Prompt: {prompt} ---")
        print(response)

asyncio.run(run_creative_writing())