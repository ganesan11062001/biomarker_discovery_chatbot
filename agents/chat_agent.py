from core.state import BiomarkerState
from agents.base_agent import BaseAgent
from config.settings import get_settings

settings = get_settings()

class ChatAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            deployment_name=settings.azure_deployment_chat,
            system_prompt_path="prompts/chat_agent.txt"
        )

    def detect_intent(self, user_query: str) -> str:
        query_lower = user_query.lower()
        if any(w in query_lower for w in ["upload", "load", "file", "csv", "excel", "data"]):
            return "ingestion_agent"
        if any(w in query_lower for w in ["qc", "quality", "missing", "outlier", "cv", "filter"]):
            return "biomarker_agent"
        if any(w in query_lower for w in ["analyze", "differential", "dea", "biomarker", "protein", "compare", "expression", "limma", "msstats", "dep"]):
            return "biomarker_agent"
        if any(w in query_lower for w in ["pathway", "enrich", "kegg", "go term", "gsea", "cluster", "ontology"]):
            return "enrichment_agent"
        if any(w in query_lower for w in ["plot", "visualize", "chart", "volcano", "heatmap", "report", "summary", "rank"]):
            return "visualization_agent"
        return "chat_agent"

    def run(self, state: BiomarkerState) -> BiomarkerState:
        user_query = state["user_query"]
        intent = self.detect_intent(user_query)
        messages = [
            {"role": "system", "content": self.system_prompt},
            *[{"role": m["role"], "content": m["content"]} for m in state["messages"]],
            {"role": "user", "content": user_query}
        ]
        response = self._call_llm(messages)
        state["messages"].append({"role": "user", "content": user_query})
        state["messages"].append({"role": "assistant", "content": response})
        state["intent"] = intent
        state["active_agent"] = intent
        state["status"] = "routed"
        return state
