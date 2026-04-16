#Example with context and structured output

import requests
import dotenv

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

dotenv.load_dotenv()


@dataclass
class Context:
    user_id: str


@dataclass
class ResponseFormat:
    summary: str
    temperature_celcius: str
    temperature_farenheit: str
    humidity: str


@tool('get_weather', description="Tool to get weather of the given city")
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()


@tool('locate_user', description="look up user's city based on the context")
def locate_user(runtime: ToolRuntime[Context]) -> str:
    match runtime.context.user_id:
        case 'ABC':
            return 'Delhi'
        case 'ZAY':
            return 'Jaipur'
        case _:
            return None


agent = create_agent(
    model = init_chat_model(model='gpt-4.1-mini', temperature=.3),
    tools = [get_weather, locate_user],
    system_prompt = "You are a helpful weather assistant, who helps with weather information and you" + \
    "are funny too",
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=InMemorySaver(),
)

agent_response = agent.invoke(
    {
        'messages': [
            {'role': 'user', 'content': 'What is the weather?'}
        ]
    },
    config={'configurable': {'thread_id': '1'}},
    context=Context(user_id="ABC"),
)

print(agent_response['structured_response'])
print(f"HUMIDITY: {agent_response['structured_response'].humidity}")
