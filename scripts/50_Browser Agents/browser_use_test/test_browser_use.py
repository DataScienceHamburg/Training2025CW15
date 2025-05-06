#%% packages
import asyncio
from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig, BrowserContextConfig
load_dotenv(find_dotenv(usecwd=True))
#%%
# def run_agent(task: str, url: str, model_provider: str = 'groq', use_vision: bool = True):
# 	if model_provider == 'groq':
# 		llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.0)
# 	else:
# 		llm = ChatOpenAI(model='gpt-4o', temperature=0.0)
# 	browser = Browser(
# 		config=BrowserConfig(
# 			headless=True,

# 			new_context_config=BrowserContextConfig(
# 				no_viewport=False,

# 			),
# 		),
# 	)
# 	browser.goto(url)
# 	agent = Agent(task=task, llm=llm, browser=browser, use_vision=use_vision, save_conversation_path='conversation.txt')
# 	asyncio.run(agent.run())

async def main():
    agent = Agent(
        task="Compare the price of gpt-4o and DeepSeek-V3",
        llm=ChatOpenAI(model="gpt-4o"),
    )
    await agent.run()

asyncio.run(main())

#%%
if __name__ == '__main__':
	main()
	
