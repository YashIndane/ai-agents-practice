"""An app for a simple Agentic chatbot for general vehicle information, using OpenAI Agents SDK"""

# imports

from agents import (
    Agent, 
    Runner,
    trace,
    function_tool,
    input_guardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper
)

from pydantic import BaseModel
from dotenv import load_dotenv

import asyncio
import os
import json
import requests
import xmltodict
import gradio as gr


# The LLM model for the agents
_MODEL = "gpt-4.1-nano"


# Defining guard pydantic schema
class GrScehma(BaseModel):
    is_registration_year_in_message: bool
    registration_year: str


# Defining gaurd rail agent
reg_year_gr_agent: Agent = Agent(
    name="Gaurd Rail Agent",
    instructions="You are an gaurd rail agent, who will detect if the user is asking \
for registration year",
    output_type=GrScehma,
    model=_MODEL
)


# Defining the gaurd rail function
@input_guardrail
async def reg_year_gr_func(
    ctx: RunContextWrapper[None],
    agent: Agent,
    message: str
) -> GuardrailFunctionOutput:
    """Function for the validating presence of regisration year in user message."""

    gr_result = await Runner.run(reg_year_gr_agent, message, context=ctx.context)
    is_reg_year_in_msg = gr_result.final_output.is_registration_year_in_message

    return GuardrailFunctionOutput(
        output_info="Found reg year query...",
        tripwire_triggered=is_reg_year_in_msg
    )
    

@function_tool
def notify_admin(reg_num: str) -> dict[str, str]:
    """Send a pushover notification to admin, when unable to fetch details from registration number"""

    PUSHOVER_URL = "https://api.pushover.net/1/messages.json"
    payload = {
        "user": os.getenv('PUSHOVER_USER'),
        "token": os.getenv('PUSHOVER_TOKEN'), 
        "message": f"Could not identify requested details for registration number: {reg_num}"
    }

    requests.post(PUSHOVER_URL, data=payload)
    return {"notified": "True"}


@function_tool
def fetch_car_details(reg_num: str) -> dict[str, str]:
    """Fetch car details from regcheck.org API and provide to user"""
    REGCHECK_URL = "http://www.regcheck.org.uk/api/reg.asmx/"

    req = requests.get(
        f"{REGCHECK_URL}Check?RegistrationNumber={reg_num}&username={os.getenv('REGCHECK_USER')}"
    )

    # response parsing
    data = xmltodict.parse(req.content)
    jdata = json.dumps(data)
    df = json.loads(jdata)
    final_data = json.loads(df["Vehicle"]["vehicleJson"])
    
    return {
        "description": final_data["Description"],
        "registration-year": final_data["RegistrationYear"],
        "engine-size": final_data["EngineSize"]["CurrentTextValue"],
        #"number-of-seats": final_data["NumberOfSeats"]["CurrentTextValue"],
        #"vehicle-identification-number":final_data["VechileIdentificationNumber"],
        "engine-number": final_data["EngineNumber"],
        "fuel-type": final_data["FuelType"]["CurrentTextValue"]
        #"registration-date": final_data["RegistrationDate"],
        #"location": final_data["Location"]
    }


# Defining Agents
data_fetch_agent: Agent = Agent(
    name="data_fetch_agent",
    instructions="You are an data fetcher agent that will fetch details \
from regcheck.org, use your fetch_car_details tool, If you can't find the details please use your notify_admin tool",
    tools=[fetch_car_details, notify_admin],
    model=_MODEL,
    handoff_description="Fetch registration number details",
)


manager_agent: Agent = Agent(
    name="manager_agent",
    instructions="You own are seasoned Car details expert, who will be helping users on \
general vehical related quieries, try to keep the conversation smooth and engaging.",
    model=_MODEL,
    handoffs=[data_fetch_agent],
    input_guardrails=[reg_year_gr_func]
)


async def chat(message: str, history) -> str:
    with trace("Vehical Info Assist"):
        try:
            result = await Runner.run(manager_agent, message)
            return result.final_output
        except InputGuardrailTripwireTriggered:
            result = "Sorry, can't answer that, but please proceed with other queries..."
            return result

if __name__ == "__main__":
    load_dotenv(override=True)
    gr.ChatInterface(chat, type="messages").launch()
