import subprocess

from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    server_ip: str = Field(
        ...,
        description="The IP of the Ubuntu server on which the ansible playbook will be run.",
    )

class MyCustomTool(BaseTool):
    name: str = "ansible_playbook_runner"
    description: str = (
        "This tool is used to run an Ansible playbook"
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, server_ip: str) -> str:
        # Implementation goes here

        result = subprocess.getstatusoutput(
            f'ansible-playbook -i {server_ip}, --private-key=~/.ssh/my-key-1.pem -u ubuntu playbook.yaml' 
        )

        print(result)
