#Simple Example

import requests
import dotenv

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool

dotenv.load_dotenv()

@tool('get_weather', description="Tool to get weather of the given city")
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

agent = create_agent(
    model = 'gpt-4.1-mini',
    tools = [get_weather],
    system_prompt = "You are a helpful weather assistant, who helps with weather information and you" + \
    "are funny too",
)

agent_response = agent.invoke(
    {
        'messages': [
            {'role': 'user', 'content': 'What is the weather like in Kochi?'}
        ]
    }
)

print(agent_response, agent_response['messages'][-1].content, end="\n")
