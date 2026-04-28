"""
agents/chat_agent.py
Entry point — uses LLM to understand user intent and routes to the right specialist.
"""
from core.state import BiomarkerState
from agents.base_agent import BaseAgent
from config.settings import get_settings

settings = get_settings()

_VALID_INTENTS = frozenset({
    "ingestion_agent",
    "biomarker_agent",
    "enrichment_agent",
    "visualization_agent",
    "chat_agent",
})

_INTENT_SYSTEM_PROMPT = """\
You are an intent classifier for a proteomics biomarker discovery chatbot.
Classify the user message into exactly one of these intents:

  ingestion_agent     — user wants to upload, load, or import a new data file
  biomarker_agent     — user explicitly wants to RUN or START a biomarker/proteomic analysis
  enrichment_agent    — user wants to RUN pathway / KEGG / GO enrichment analysis
  visualization_agent — user wants to generate plots, heatmaps, charts, or a report
  chat_agent          — everything else: questions, explanations, general conversation, off-topic

Rules:
- Questions ("what is X?", "explain X", "how does X work?", "tell me about X") → chat_agent
- Off-topic messages ("what is random forest", general ML/statistics questions) → chat_agent
- Asking about existing results ("show my top proteins", "what did the analysis find?") → chat_agent
- Only route to biomarker_agent when the user explicitly says to RUN or START analysis
- Only route to enrichment_agent / visualization_agent when user says to RUN those steps

Reply with ONLY the intent name — no explanation, no punctuation, nothing else.\
"""


class ChatAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            deployment_name=settings.azure_deployment_chat,
            system_prompt_path="prompts/chat_agent.txt",
        )

    # ── Intent detection (LLM-based) ──────────────────────────────────────────

    def detect_intent(self, query: str, state: BiomarkerState) -> str:
        """Ask the LLM to classify the user's intent; fall back to chat_agent on failure."""
        data_loaded   = bool(state.get("data_type"))
        analysis_done = state.get("n_significant") is not None

        context = (
            f"Session context:\n"
            f"  Data loaded: {data_loaded}\n"
            f"  Analysis already complete: {analysis_done}\n"
            f"  Current status: {state.get('status', 'ready')}\n\n"
            f'User message: "{query}"\n\n'
            f"Intent:"
        )

        messages = [
            {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
            {"role": "user",   "content": context},
        ]

        try:
            raw = self._call_llm(messages, max_tokens=15, temperature=0.0).strip().lower()
            for intent in _VALID_INTENTS:
                if intent in raw:
                    self.logger.debug("LLM intent: %r (raw=%r)", intent, raw)
                    return intent
        except Exception as exc:
            self.logger.warning("Intent LLM call failed (%s) — defaulting to chat_agent.", exc)

        return "chat_agent"

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, state: BiomarkerState) -> BiomarkerState:
        user_query = state.get("user_query", "")
        intent     = self.detect_intent(user_query, state)

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
