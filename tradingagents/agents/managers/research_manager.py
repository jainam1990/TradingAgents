"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

import logging

from tradingagents.agents.schemas import PortfolioRating, ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)

logger = logging.getLogger(__name__)

# Maps conviction buckets → allowed rating tiers.
# Any recommendation outside the allowed set is overridden to the boundary tier.
_CONVICTION_GATE: list[tuple[float, float, list[PortfolioRating], PortfolioRating]] = [
    # (low, high, allowed_ratings, override_toward)
    (0.00, 0.35, [PortfolioRating.SELL, PortfolioRating.UNDERWEIGHT], PortfolioRating.UNDERWEIGHT),
    (0.35, 0.45, [PortfolioRating.UNDERWEIGHT, PortfolioRating.HOLD], PortfolioRating.UNDERWEIGHT),
    (0.45, 0.55, [PortfolioRating.HOLD], PortfolioRating.HOLD),
    (0.55, 0.65, [PortfolioRating.OVERWEIGHT, PortfolioRating.HOLD], PortfolioRating.OVERWEIGHT),
    (0.65, 1.01, [PortfolioRating.BUY, PortfolioRating.OVERWEIGHT], PortfolioRating.OVERWEIGHT),
]


def _apply_conviction_gate(plan: ResearchPlan) -> ResearchPlan:
    """Override recommendation if it deviates from the conviction score bucket."""
    conviction = plan.bull_conviction
    for low, high, allowed, override in _CONVICTION_GATE:
        if low <= conviction < high:
            if plan.recommendation not in allowed:
                logger.warning(
                    "Conviction gate: bull_conviction=%.2f but recommendation=%s "
                    "(allowed: %s) — overriding to %s",
                    conviction,
                    plan.recommendation.value,
                    [r.value for r in allowed],
                    override.value,
                )
                return plan.model_copy(update={"recommendation": override})
            break
    return plan


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]

        prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction in the bull thesis; recommend taking or growing the position
- **Overweight**: Constructive view; recommend gradually increasing exposure
- **Hold**: Balanced view; recommend maintaining the current position
- **Underweight**: Cautious view; recommend trimming exposure
- **Sell**: Strong conviction in the bear thesis; recommend exiting or avoiding the position

Commit to a clear stance whenever the debate's strongest arguments warrant one; reserve Hold for situations where the evidence on both sides is genuinely balanced.

---

**Debate History:**
{history}""" + get_language_instruction()

        raw_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
            return_parsed=True,
        )

        # Apply conviction gate if we got a structured ResearchPlan back.
        # When the provider fell back to free-text, raw_plan is already a string.
        if isinstance(raw_plan, ResearchPlan):
            gated = _apply_conviction_gate(raw_plan)
            investment_plan = render_research_plan(gated)
        else:
            investment_plan = raw_plan

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
