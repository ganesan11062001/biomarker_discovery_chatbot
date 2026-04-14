"""
agents/chat_agent.py
Entry point — detects user intent and routes to the right specialist.
"""
from core.state import BiomarkerState
from agents.base_agent import BaseAgent
from config.settings import get_settings

settings = get_settings()


class ChatAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            deployment_name=settings.azure_deployment_chat,
            system_prompt_path="prompts/chat_agent.txt",
        )

    # ── Intent detection ──────────────────────────────────────────────────────

    def detect_intent(self, query: str) -> str:
        q = query.lower()

        if any(w in q for w in [
            "upload", "load", "file", "csv", "excel", "data", "import",
        ]):
            return "ingestion_agent"

        if any(w in q for w in [
            "analyz", "run", "start", "biomarker", "differential",
            "dea", "compare", "find", "identify", "discover",
            "significant", "expression", "protein", "fold change",
        ]):
            # Only route to biomarker agent if data is already loaded
            return "biomarker_agent"

        if any(w in q for w in [
            "pathway", "enrich", "kegg", "go term", "gsea",
            "cluster", "ontology",
        ]):
            return "enrichment_agent"

        if any(w in q for w in [
            "plot", "visualize", "visualise", "chart",
            "volcano", "heatmap", "report",
        ]):
            return "visualization_agent"

        return "chat_agent"

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, state: BiomarkerState) -> BiomarkerState:
        user_query = state.get("user_query", "")
        intent     = self.detect_intent(user_query)

        messages_for_llm = [
            {"role": "system", "content": self.system_prompt},
            *[
                {"role": m["role"], "content": m["content"]}
                for m in (state.get("messages") or [])
                if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            ],
            {"role": "user", "content": user_query},
        ]

        response = self._call_llm(messages_for_llm)

        state["messages"].append({"role": "user",      "content": user_query})
        state["messages"].append({"role": "assistant",  "content": response})
        state["intent"]       = intent
        state["active_agent"] = intent
        state["status"]       = "routed"
        return state
