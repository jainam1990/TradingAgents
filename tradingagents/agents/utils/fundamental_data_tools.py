from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_fundamentals(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing comprehensive fundamental data
    """
    return route_to_vendor("get_fundamentals", ticker, curr_date)


@tool
def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve balance sheet data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing balance sheet data
    """
    return route_to_vendor("get_balance_sheet", ticker, freq, curr_date)


@tool
def get_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve cash flow statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing cash flow statement data
    """
    return route_to_vendor("get_cashflow", ticker, freq, curr_date)


@tool
def get_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve income statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing income statement data
    """
    return route_to_vendor("get_income_statement", ticker, freq, curr_date)


@tool
def calculate_dcf_fair_value(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    growth_rate: Annotated[float, "near-term FCF growth rate as decimal, e.g. 0.10 for 10%"] = 0.10,
    terminal_growth_rate: Annotated[float, "perpetual terminal growth rate as decimal, e.g. 0.03"] = 0.03,
    discount_rate: Annotated[float, "WACC / discount rate as decimal, e.g. 0.10"] = 0.10,
    projection_years: Annotated[int, "number of years to project FCF, 5 or 10"] = 5,
) -> str:
    """
    Compute a simple DCF fair-value estimate for a stock using the most recent
    trailing twelve-month (TTM) free cash flow pulled from yfinance.

    Free Cash Flow = Operating Cash Flow - Capital Expenditures (TTM).
    Projects FCF forward at `growth_rate` for `projection_years`, then applies
    a Gordon-Growth terminal value, discounts everything at `discount_rate`, and
    divides by shares outstanding to get per-share intrinsic value.

    Args:
        ticker: Ticker symbol
        curr_date: Trade date (yyyy-mm-dd) — used to avoid look-ahead bias
        growth_rate: Near-term annual FCF growth rate (default 10%)
        terminal_growth_rate: Perpetual growth rate after projection period (default 3%)
        discount_rate: WACC used for discounting (default 10%)
        projection_years: Forecast horizon in years (default 5)

    Returns:
        Formatted string with assumptions, projected FCFs, and per-share fair value.
    """
    try:
        import yfinance as yf
        import datetime

        cutoff = datetime.date.fromisoformat(curr_date)
        tk = yf.Ticker(ticker)

        # TTM operating cash flow and capex from annual cash flow statement
        cf = tk.cashflow
        if cf is None or cf.empty:
            return f"DCF: No cash flow data available for {ticker}."

        # Filter columns (fiscal year-end dates) to those on or before curr_date
        valid_cols = [c for c in cf.columns if hasattr(c, 'date') and c.date() <= cutoff]
        if not valid_cols:
            valid_cols = list(cf.columns[:1])  # fallback: most recent available

        col = valid_cols[0]

        def _get(row_key):
            for key in cf.index:
                if row_key.lower() in str(key).lower():
                    val = cf.loc[key, col]
                    if val is not None and str(val) not in ("nan", "None"):
                        return float(val)
            return None

        op_cf = _get("operating")
        capex = _get("capital expenditure")
        if op_cf is None:
            return f"DCF: Could not find operating cash flow for {ticker}."
        capex = capex or 0.0
        fcf = op_cf - abs(capex)

        # Shares outstanding
        info = tk.info or {}
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        if not shares:
            return f"DCF: Could not retrieve shares outstanding for {ticker}."

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")

        # Project FCFs
        projected = []
        pv_sum = 0.0
        for yr in range(1, projection_years + 1):
            future_fcf = fcf * ((1 + growth_rate) ** yr)
            pv = future_fcf / ((1 + discount_rate) ** yr)
            projected.append((yr, future_fcf, pv))
            pv_sum += pv

        # Terminal value (Gordon Growth)
        terminal_fcf = fcf * ((1 + growth_rate) ** projection_years) * (1 + terminal_growth_rate)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
        pv_terminal = terminal_value / ((1 + discount_rate) ** projection_years)

        total_pv = pv_sum + pv_terminal
        fair_value_per_share = total_pv / shares

        lines = [
            f"=== DCF Fair Value: {ticker} (as of {curr_date}) ===",
            f"TTM Free Cash Flow : ${fcf/1e9:.2f}B  (OpCF ${op_cf/1e9:.2f}B - CapEx ${abs(capex)/1e9:.2f}B)",
            f"Assumptions        : growth={growth_rate*100:.1f}% / {projection_years}yr, "
            f"terminal={terminal_growth_rate*100:.1f}%, WACC={discount_rate*100:.1f}%",
            "",
            "Projected FCFs (discounted):",
        ]
        for yr, fcf_yr, pv_yr in projected:
            lines.append(f"  Year {yr}: ${fcf_yr/1e9:.2f}B FCF → PV ${pv_yr/1e9:.2f}B")
        lines += [
            f"  Terminal value PV : ${pv_terminal/1e9:.2f}B",
            f"  Total equity value: ${total_pv/1e9:.2f}B",
            f"",
            f"Fair Value / Share : ${fair_value_per_share:.2f}",
        ]
        if current_price:
            premium = (current_price - fair_value_per_share) / fair_value_per_share * 100
            sign = "+" if premium >= 0 else ""
            lines.append(f"Current Price      : ${current_price:.2f}  ({sign}{premium:.1f}% vs fair value)")

        return "\n".join(lines)

    except Exception as exc:
        return f"DCF calculation failed for {ticker}: {exc}"
