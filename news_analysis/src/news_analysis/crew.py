#%%
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, WebsiteSearchTool
from langchain_community.tools import DuckDuckGoSearchRun, arxiv
from crewai.tools import tool
import time


#%%
from pydantic import BaseModel
# uv add duckduckgo-search
# from langchain_community.tools import DuckDuckGoSearchRun

@tool("Arxive")
def arxive_search(tool_input: str):
    """Searches the arxive.org for search_query."""
    search = arxiv.ArxivQueryRun()
    res = search.invoke(input=tool_input)
    return res

@tool("DuckDuck")
def duck_duck_search(tool_input: str):
    """Searches the internet for search_query."""
    search = DuckDuckGoSearchRun()
    res = search.invoke(input=tool_input)
    return res

tools = [WebsiteSearchTool(), duck_duck_search, arxive_search]
#%%

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

class MyOutputFormat(BaseModel):
    chapter_title: str
    bullet_points: list[str]

@CrewBase
class NewsAnalysis():
    """NewsAnalysis crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            tools=tools,
            max_rpm=10  # Rate limit: maximum 10 requests per minute
            
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True,
            max_rpm=10  # Rate limit: maximum 10 requests per minute
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            # output_pydantic=MyOutputFormat,
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the NewsAnalysis crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
