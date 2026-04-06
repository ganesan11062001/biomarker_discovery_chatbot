import subprocess
import json
from abc import ABC, abstractmethod
from pathlib import Path

class BaseSkill(ABC):
    def __init__(self, script_path: str):
        self.script_path = Path(script_path)

    def run_r_script(self, args: dict) -> dict:
        args_json = json.dumps(args)
        result = subprocess.run(
            ["Rscript", str(self.script_path), args_json],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"R script failed: {result.stderr}")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"raw_output": result.stdout}

    def run_python_script(self, args: dict) -> dict:
        import importlib.util
        spec = importlib.util.spec_from_file_location("skill", self.script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.run(args)

    @abstractmethod
    def execute(self, **kwargs) -> dict:
        pass
