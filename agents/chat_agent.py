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

        # Ingestion: user explicitly wants to load a NEW file.
        # Use specific phrases so "uploaded dataset" doesn't trigger this.
        if any(w in q for w in [
            "upload a", "upload my", "upload new",
            "load a file", "load my file", "load new",
            "import file", "import data",
            "new file", "open file", "attach file",
            ".csv", ".xlsx", ".xls",
        ]):
            return "ingestion_agent"

        # Analysis: user wants to run or trigger analysis.
        # Note: "analyz" catches "analyze/analyzing" but NOT "analysis" (ends in -sis).
        # Add "analysis" and "proteomic" explicitly.
        if any(w in q for w in [
            "run analysis", "run fold", "run differential",
            "do the analysis", "do analysis", "start analysis",
            "perform analysis", "perform the",
            "fold-change", "fold change",
            "analyz", "analysis",
            "proteomic", "proteomics",
            "biomarker", "differential expression",
            "dea", "compare groups", "find significant",
            "identify biomarker", "discover biomarker",
            "top protein", "significant protein",
        ]):
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

        # Always record the user turn first
        state["messages"].append({"role": "user", "content": user_query})

        if intent == "chat_agent":
            # Build a context block so the LLM knows what data is loaded.
            context_lines = []
            if state.get("data_type"):
                context_lines += [
                    "## Current session state",
                    f"- Data loaded: YES",
                    f"- Proteins: {state.get('n_proteins', '?')}",
                    f"- Samples: {state.get('n_samples', '?')}",
                    f"- Data type: {state.get('data_type', '?')}",
                    f"- Omic type: {state.get('omic_type', 'proteomics')}",
                    f"- Pooled design: {state.get('is_pooled_design', False)}",
                    f"- Groups: {state.get('label_map') or 'not assigned'}",
                    f"- Pipeline status: {state.get('status', 'data_loaded')}",
                    f"- Analysis complete: {bool(state.get('n_significant'))}",
                ]
            else:
                context_lines += [
                    "## Current session state",
                    "- Data loaded: NO — user has not uploaded a file yet.",
                ]

            system_with_context = (
                self.system_prompt
                + "\n\n"
                + "\n".join(context_lines)
                + "\n\nAlways answer based on the actual session state above. "
                  "Never say data has not been uploaded if 'Data loaded: YES' is shown."
            )

            # Cap history to last 10 turns to avoid token creep
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in (state.get("messages") or [])
                if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            ][-10:]

            messages_for_llm = [
                {"role": "system", "content": system_with_context},
                *history,
            ]
            response = self._call_llm(messages_for_llm)
            state["messages"].append({"role": "assistant", "content": response})
        # When routing to a specialist agent, skip the LLM call here.
        # The specialist owns its own response and appends it to state["messages"].

        state["intent"]       = intent
        state["active_agent"] = intent
        state["status"]       = "routed"
        return state
