from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from ansible_devops_crew.tools.custom_tool import MyCustomTool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class AnsibleDevopsCrew():
    """AnsibleDevopsCrew crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def team_lead(self) -> Agent:
        return Agent(
            config=self.agents_config['team_lead'], # type: ignore[index]
            verbose=True,
        )

    @agent
    def devops_engg(self) -> Agent:
        return Agent(
            config=self.agents_config['devops_engg'], # type: ignore[index]
            tools=[MyCustomTool()],
            verbose=True,
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['analysis_task'], # type: ignore[index]
        )

    @task
    def development_task(self) -> Task:
        return Task(
            config=self.tasks_config['development_task'], # type: ignore[index]
            output_file='playbook.yaml',
        )

    @task
    def ansible_run_task(self) -> Task:
        return Task(
            config=self.tasks_config['ansible_run_task'],
       )

    @crew
    def crew(self) -> Crew:
        """Creates the AnsibleDevopsCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
