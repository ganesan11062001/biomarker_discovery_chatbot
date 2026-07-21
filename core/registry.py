from typing import Dict
from agents.base_agent import BaseAgent
from skills.base_skill import BaseSkill

class AgentRegistry:
    _agents: Dict[str, BaseAgent] = {}

    @classmethod
    def register(cls, name: str, agent: BaseAgent):
        cls._agents[name] = agent

    @classmethod
    def get(cls, name: str) -> BaseAgent:
        if name not in cls._agents:
            raise KeyError(f"Agent '{name}' not registered.")
        return cls._agents[name]

    @classmethod
    def list_agents(cls) -> list:
        return list(cls._agents.keys())

class SkillRegistry:
    _skills: Dict[str, BaseSkill] = {}

    @classmethod
    def register(cls, name: str, skill: BaseSkill):
        cls._skills[name] = skill

    @classmethod
    def get(cls, name: str) -> BaseSkill:
        if name not in cls._skills:
            raise KeyError(f"Skill '{name}' not registered.")
        return cls._skills[name]

    @classmethod
    def list_skills(cls) -> list:
        return list(cls._skills.keys())
