
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from metrics.token_tracker import TokenTrackingCallback

from . import inventory
from .base_agent import BaseAgent


class TravelIntent(BaseModel):
    destination: str = Field(description="Primary travel destination or region")
    origin: str | None = Field(None, description="Departure city/airport")
    budget_usd: float = Field(description="Total budget in USD")
    duration_days: int | None = Field(None, description="Trip duration in days")
    travel_month: str | None = Field(None, description="Preferred travel month")
    travel_year: int | None = Field(None, description="Preferred travel year")
    interests: list[str] = Field(default_factory=list, description="Traveller interests e.g. beaches, food, history")
    group_size: int = Field(default=1, description="Number of travellers")
    accommodation_preference: str | None = Field(None, description="hotel, hostel, airbnb, resort")
    flexibility: str = Field(default="moderate", description="low | moderate | high — how flexible dates/budget are")
    raw_query: str = Field(description="Original user query verbatim")


SYSTEM_PROMPT = """You are an expert travel intent parser.
Extract structured travel requirements from the user's message.
Be precise with budget — if they say "under $2000" set budget_usd to 2000.
If month is mentioned without a year, use the next occurrence. Today's date context will be provided.
For interests, extract specific terms: beaches, local food, history, nightlife, hiking, culture, etc.
Return ONLY valid JSON matching the schema — no explanation."""

HUMAN_PROMPT = """Today's date: {today}

User message: {query}

Extract travel intent as JSON."""


class IntentParserAgent(BaseAgent):
    name = "intent_parser"
    description = "Parses raw user query into structured TravelIntent"

    def _setup(self):
        try:
            from config import effective_temperature, get_api_config
            model = get_api_config().llm.intent_parser_model
        except Exception:
            model = "claude-sonnet-4-6"
        self.llm = ChatAnthropic(
            model=model,
            temperature=effective_temperature(0),
            callbacks=[TokenTrackingCallback(model=model)],
        ).with_structured_output(TravelIntent)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self.chain = self.prompt | self.llm

    async def _execute(self, state: dict) -> dict:
        from datetime import date
        query = state.get("user_query", "")
        if not query:
            return self._error_state("No user_query in state")

        async def _parse() -> dict:
            intent: TravelIntent = await self.chain.ainvoke({
                "today": date.today().isoformat(),
                "query": query,
            })
            return {"intent": intent.model_dump()}

        from config.settings import get_settings
        if get_settings().inventory_mode in ("capture", "replay"):
            # Parsed intent feeds every downstream query id and prompt. Record it
            # per raw query so a replay starts from the same structured intent
            # instead of re-parsing (which can vary, and depends on today's date).
            query_id = inventory.inventory_query_id("intent", query=query)
            parsed = await inventory.replay_or_record(
                "intent", query_id, _parse, run_label="intent-capture",
            )
        else:
            parsed = await _parse()

        return {"intent": parsed["intent"], "status": "intent_parsed"}
