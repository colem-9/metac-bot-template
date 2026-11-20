import argparse
import asyncio
import logging
from datetime import datetime, timezone
from typing import Literal

import dotenv

# Runtime helpers (env validation, banners, dependency-warning suppression).
from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class FallTemplateBot2026(ForecastBot):
    """
    This is the template bot for the Fall 2026 FutureEval Bot Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    This bot is identical to SummerTemplateBot2026.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name")
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 1

    ##################################### RESEARCH #####################################

    ############################ OWL360 UPGRADES ############################
    # Everything in this block is custom on top of the Metaculus template:
    #   1. Multi-angle research (news + base rates + resolution status) run in parallel
    #   2. Optional AskNews as a second news source (used automatically if keys are set)
    #   3. Live prediction-market odds (Polymarket + Manifold) as a research input
    #   4. A "persona ensemble": each of the N forecasts reasons from a different angle
    #   5. A countdown/timing brief so the model reasons about time left
    #   6. Smarter aggregation: geometric mean of odds, then cap at 3-97%
    #   7. Statistical simulation model for numeric questions about market/economic data series

    # Final binary forecasts are clamped to this range.
    BINARY_FLOOR = 0.03
    BINARY_CEILING = 0.97
    # Each multiple-choice option keeps at least this much probability.
    MC_FLOOR = 0.01
    # Turn individual pieces on/off without touching the code below.
    USE_MARKET_ODDS = True
    USE_BASE_RATE_RESEARCH = True
    USE_PERSONAS = True

    PERSONAS = [
        (
            "Base-rate statistician",
            "Start from the outside view. Name a reference class of similar past situations, "
            "estimate how often outcomes like this happened, and only then adjust for the specifics "
            "of this case. Keep adjustments modest unless the evidence is strong.",
        ),
        (
            "News-driven analyst",
            "Focus on the most recent concrete developments and their momentum. Ask what has to "
            "happen, by whom, and by when for each outcome, and whether that timeline is realistic.",
        ),
        (
            "Red-team skeptic",
            "Your job is to find why the obvious answer could be wrong. Look for overlooked "
            "scenarios, overconfidence in the research, and technicalities in the resolution "
            "criteria or fine print that could flip the outcome.",
        ),
        (
            "Market-aware trader",
            "Treat any prediction-market prices in the research as a strong prior from people with "
            "money at stake. If no market matches this exact question, anchor on the status quo. "
            "Only deviate from a matching market when you can name the specific reason it is wrong "
            "(different resolution criteria, different deadline, stale or thin market).",
        ),
        (
            "Resolution-criteria lawyer",
            "Read the resolution criteria and fine print literally. Decide exactly what evidence "
            "would count, which source resolves it, and whether edge cases (partial events, delays, "
            "ambiguous wording) push the answer one way. Forecast what the resolver will rule, not "
            "what feels true.",
        ),
    ]

    _persona_counters: dict = {}

    def _next_persona(self, question: MetaculusQuestion) -> tuple[str, str]:
        key = question.id_of_question or question.page_url
        index = self._persona_counters.get(key, 0)
        self._persona_counters[key] = index + 1
        return self.PERSONAS[index % len(self.PERSONAS)]

    @staticmethod
    def _timing_brief(question: MetaculusQuestion) -> str:
        now = datetime.now(timezone.utc)
        lines = [f"Today (UTC): {now.strftime('%Y-%m-%d')}"]
        for label, value in (
            ("Question closes", question.close_time),
            ("Scheduled resolution", question.scheduled_resolution_time),
        ):
            if value is None:
                continue
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            days = (value - now).total_seconds() / 86400
            lines.append(f"{label}: {value.strftime('%Y-%m-%d')} ({days:.1f} days from now)")
        lines.append(
            "Short windows favor the status quo: if the event has not happened yet and little "
            "time is left, it usually will not happen."
        )
        return "\n".join(lines)

    async def _invoke_forecaster(self, question: MetaculusQuestion, prompt: str) -> str:
        preamble = ["TIMING:\n" + self._timing_brief(question)]
        if self.USE_PERSONAS:
            name, instructions = self._next_persona(question)
            preamble.append(
                f"YOUR ROLE ON THIS TEAM: {name}.\n{instructions}\n"
                "Other forecasters on the team take different angles; your forecast will be "
                "combined with theirs, so give your honest independent view."
            )
        full_prompt = "\n\n".join(preamble) + "\n\n" + prompt
        reasoning = await self.get_llm("default", "llm").invoke(full_prompt)
        if self.USE_PERSONAS:
            reasoning = f"[Perspective: {name}]\n{reasoning}"
        return reasoning

    # ---------- research ----------

    def _research_header(self, question: MetaculusQuestion) -> str:
        return clean_indents(
            f"""
            Question:
            {question.question_text}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}
            """
        )

    def _news_research_prompt(self, question: MetaculusQuestion) -> str:
        return (
            "You are an assistant to a superforecaster. Give a concise but detailed rundown of the "
            "most relevant RECENT news (with dates) for the question below, and say whether it "
            "would resolve Yes or No based on current information. Do not produce a forecast.\n\n"
            + self._research_header(question)
        )

    def _base_rate_research_prompt(self, question: MetaculusQuestion) -> str:
        return (
            "You are an assistant to a superforecaster. Do NOT summarize today's news. Instead:\n"
            "1. Identify 1-3 reference classes of comparable past events and give concrete base "
            "rates with numbers and sources where possible.\n"
            "2. Identify the official resolution source named in the criteria and report its "
            "current value/status as precisely as you can (latest data point and date).\n"
            "3. List any scheduled events before the deadline (votes, data releases, earnings, "
            "court dates, launches) that could decide the outcome.\n"
            "Do not produce a forecast.\n\n" + self._research_header(question)
        )

    async def _safe(self, label: str, coro) -> str:
        try:
            result = await coro
            return result or ""
        except Exception as e:  # one failing source should never sink the question
            logger.warning(f"Research source '{label}' failed: {e}")
            return ""

    async def _asknews_research(self, question: MetaculusQuestion) -> str:
        import os

        if not (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")):
            return ""
        return await AskNewsSearcher().call_preconfigured_version(
            "asknews/news-summaries", question.question_text
        )

    async def _market_search_query(self, question: MetaculusQuestion) -> str:
        prompt = (
            "Write a short search query (3-6 words, no quotes, no punctuation) that would find a "
            "prediction market about this question on Polymarket. Output only the query.\n\n"
            f"Question: {question.question_text}"
        )
        query = await self.get_llm("summarizer", "llm").invoke(prompt)
        return query.strip().strip('"').splitlines()[0][:80]

    async def _market_odds_research(self, question: MetaculusQuestion) -> str:
        import json

        import httpx

        query = await self._market_search_query(question)
        lines: list[str] = []
        async with httpx.AsyncClient(timeout=15) as client:
            # Polymarket (real money)
            try:
                r = await client.get(
                    "https://gamma-api.polymarket.com/public-search",
                    params={"q": query, "limit_per_type": 10},
                )
                r.raise_for_status()
                for event in (r.json().get("events") or [])[:10]:
                    if event.get("closed"):
                        continue
                    for m in (event.get("markets") or [])[:4]:
                        if m.get("closed"):
                            continue
                        try:
                            outcomes = json.loads(m.get("outcomes") or "[]")
                            prices = json.loads(m.get("outcomePrices") or "[]")
                        except (TypeError, ValueError):
                            continue
                        if not outcomes or len(outcomes) != len(prices):
                            continue
                        priced = ", ".join(
                            f"{o}: {float(p) * 100:.0f}%" for o, p in zip(outcomes, prices)
                        )
                        volume = float(m.get("volume") or 0)
                        end = (m.get("endDate") or event.get("endDate") or "")[:10]
                        lines.append(
                            f"- Polymarket | {m.get('question') or event.get('title')} | "
                            f"{priced} | ends {end or '?'} | volume ${volume:,.0f}"
                        )
            except Exception as e:
                logger.warning(f"Polymarket lookup failed: {e}")
            # Manifold (play money, but broad coverage)
            try:
                r = await client.get(
                    "https://api.manifold.markets/v0/search-markets",
                    params={"term": query, "filter": "open", "limit": 5},
                )
                r.raise_for_status()
                for m in r.json()[:5]:
                    if m.get("outcomeType") != "BINARY" or m.get("probability") is None:
                        continue
                    lines.append(
                        f"- Manifold (play money) | {m.get('question')} | "
                        f"Yes: {m['probability'] * 100:.0f}% | traders: {m.get('uniqueBettorCount', '?')}"
                    )
            except Exception as e:
                logger.warning(f"Manifold lookup failed: {e}")
        if not lines:
            return ""
        return (
            f"Search query used: '{query}'. These markets were found by keyword search and may NOT "
            "match this exact question. Check the wording, deadline and resolution criteria "
            "before relying on any of them; ignore ones that are about something different. "
            "Thin markets (low volume / few traders) are weak evidence.\n" + "\n".join(lines[:12])
        )

    # ---------- statistical model for numeric data-series questions ----------
    # For numeric questions about a fetchable series (stock/index/crypto/FX/commodity
    # prices via Yahoo Finance, economic data via FRED), simulate the series forward
    # from its own historical moves and hand the AI the resulting percentiles as an anchor.

    USE_SERIES_MODEL = True
    SERIES_MODEL_PATHS = 20000

    async def _identify_series(self, question: MetaculusQuestion) -> dict | None:
        import json

        prompt = clean_indents(
            f"""
            Decide whether this forecasting question asks for the future value of a public data
            series that can be downloaded, and if so identify it. Reply with ONLY a JSON object.

            Use source "yahoo" for market prices (stocks like AAPL, indices like ^GSPC ^NDX ^DJI
            ^VIX ^TNX, crypto like BTC-USD ETH-USD, FX like EURUSD=X USDJPY=X, futures like CL=F
            GC=F). Use source "fred" for US economic data with its FRED series id (e.g. UNRATE,
            CPIAUCSL, FEDFUNDS, DGS10, PAYEMS, GDP, MORTGAGE30US, DCOILWTICO).

            transform: "level" if the question asks for the series value itself; "yoy_pct" if it
            asks for the year-over-year % change of the series (e.g. CPI inflation rate).
            If the question needs any other calculation, a different statistic (max, average,
            count), or no matching series exists, set source to "none".

            target_date: the date (YYYY-MM-DD) whose value resolves the question.
            additive: true if the series is a rate/percentage that can approach zero or go
            negative (interest rates, unemployment rate, yields); false for prices and levels.

            Format: {{"source": "yahoo"|"fred"|"none", "symbol": "...", "transform": "level"|"yoy_pct",
            "target_date": "YYYY-MM-DD", "additive": true|false, "units_note": "..."}}

            Question: {question.question_text}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            Unit of measure: {getattr(question, "unit_of_measure", "") or "unspecified"}
            """
        )
        raw = await self.get_llm("summarizer", "llm").invoke(prompt)
        try:
            spec = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            return None
        if spec.get("source") not in ("yahoo", "fred") or not spec.get("symbol"):
            return None
        return spec

    @staticmethod
    async def _fetch_series(source: str, symbol: str) -> list[tuple]:
        """Returns [(date, value), ...] sorted by date."""
        import csv
        import io

        import httpx

        headers = {"User-Agent": "Mozilla/5.0 (forecasting bot; contact via Metaculus)"}
        async with httpx.AsyncClient(timeout=20, headers=headers, follow_redirects=True) as client:
            if source == "fred":
                r = await client.get(
                    "https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": symbol}
                )
                r.raise_for_status()
                rows = list(csv.reader(io.StringIO(r.text)))[1:]
                out = []
                for row in rows:
                    if len(row) < 2 or row[1] in ("", "."):
                        continue
                    out.append((datetime.strptime(row[0], "%Y-%m-%d").date(), float(row[1])))
                return out
            last_error = None
            for host in ("query1", "query2"):
                try:
                    r = await client.get(
                        f"https://{host}.finance.yahoo.com/v8/finance/chart/{symbol}",
                        params={"range": "3y", "interval": "1d"},
                    )
                    r.raise_for_status()
                    result = r.json()["chart"]["result"][0]
                    closes = result["indicators"]["quote"][0]["close"]
                    return [
                        (datetime.fromtimestamp(t, tz=timezone.utc).date(), float(c))
                        for t, c in zip(result["timestamp"], closes)
                        if c is not None
                    ]
                except Exception as e:
                    last_error = e
            raise RuntimeError(f"Yahoo fetch failed for {symbol}: {last_error}")

    @classmethod
    def _simulate_series(cls, series, target_date, additive: bool, transform: str, n_paths: int):
        """Bootstrap the series' own historical step changes forward to target_date."""
        import numpy as np

        dates = [d for d, _ in series]
        values = np.array([v for _, v in series], dtype=float)
        if len(values) < 30:
            raise ValueError("not enough history")
        gaps = np.diff([d.toordinal() for d in dates])
        step_days = float(np.median(gaps))  # ~1 for daily, ~30 for monthly, ~7 weekly
        last_date, last_value = dates[-1], values[-1]
        days_ahead = (target_date - last_date).days
        if days_ahead <= 0:
            return None  # already known or past; let the AI read the data
        if step_days <= 1.5:  # daily trading data: count weekdays
            steps = int(np.busday_count(last_date, target_date))
            lookback = 750
        else:
            steps = max(1, round(days_ahead / step_days))
            lookback = 120
        use_additive = additive or values.min() <= 0
        hist = values[-(lookback + 1):]
        moves = np.diff(hist) if use_additive else np.diff(np.log(hist))
        rng = np.random.default_rng(7)
        draws = rng.choice(moves, size=(n_paths, max(steps, 1)))
        if step_days <= 1.5:
            # daily market prices: remove drift so we don't extrapolate a past rally/selloff
            draws = draws - moves.mean()
        else:
            # economic series (CPI, payrolls, GDP...): the trend IS the signal, so keep the
            # recent trend (last ~3 years of steps) instead of the long-run one
            draws = draws - moves.mean() + moves[-36:].mean()
        total = draws.sum(axis=1)
        finals = last_value + total if use_additive else last_value * np.exp(total)
        if transform == "yoy_pct":
            year_ago_target = target_date.replace(year=target_date.year - 1)
            past = [v for d, v in series if d <= year_ago_target]
            if not past:
                raise ValueError("no value a year before target date")
            finals = (finals / past[-1] - 1) * 100
            current = (last_value / [v for d, v in series if d <= last_date.replace(year=last_date.year - 1)][-1] - 1) * 100
        else:
            current = last_value
        qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct = dict(zip(qs, np.percentile(finals, qs)))
        return {
            "last_date": last_date,
            "last_value": last_value,
            "current": current,
            "steps": steps,
            "step_days": step_days,
            "additive": use_additive,
            "percentiles": pct,
            "finals": finals,
        }

    async def _series_model_research(self, question: MetaculusQuestion) -> str:
        import numpy as np

        if not isinstance(question, NumericQuestion):
            return ""
        spec = await self._identify_series(question)
        if not spec:
            return ""
        try:
            target = datetime.strptime(spec["target_date"], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            when = question.scheduled_resolution_time or question.close_time
            if when is None:
                return ""
            target = when.date()
        series = await self._fetch_series(spec["source"], spec["symbol"])
        transform = spec.get("transform", "level")
        sim = self._simulate_series(
            series, target, bool(spec.get("additive")), transform, self.SERIES_MODEL_PATHS
        )
        if sim is None:
            return ""
        p = sim["percentiles"]
        fmt = lambda x: f"{x:,.4g}"
        lines = [
            f"Series: {spec['symbol']} ({spec['source']}), transform: {transform}. {spec.get('units_note', '')}",
            f"Latest observation: {fmt(sim['last_value'])} on {sim['last_date']}"
            + (f" (current YoY: {sim['current']:.2f}%)" if transform == "yoy_pct" else ""),
            f"Target date: {target} (~{sim['steps']} steps of ~{sim['step_days']:.0f} day(s) ahead).",
            "Model: bootstrap of this series' own historical "
            + ("changes" if sim["additive"] else "% moves")
            + f", {self.SERIES_MODEL_PATHS:,} simulated paths, "
            + ("trend removed." if sim["step_days"] <= 1.5 else "recent ~3-year trend kept."),
            "Model percentiles: "
            + ", ".join(f"P{q}={fmt(v)}" for q, v in p.items()),
        ]
        lb, ub = getattr(question, "lower_bound", None), getattr(question, "upper_bound", None)
        finals = sim["finals"]
        if lb is not None and ub is not None:
            lines.append(
                f"Model P(below question lower bound {fmt(lb)}) = {np.mean(finals < lb):.1%}; "
                f"P(above upper bound {fmt(ub)}) = {np.mean(finals > ub):.1%}."
            )
        lines.append(
            "How to use this: it is a purely statistical baseline that knows nothing about news. "
            "Keep your distribution close to it, especially its width and tails, unless the "
            "research names a specific scheduled catalyst (data release, earnings, central bank "
            "meeting, policy change) or a resolution detail that the model cannot know. Check the "
            "series matches the resolution source; if it does not, ignore this section."
        )
        return "\n".join(lines)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            researcher = self.get_llm("researcher", "llm")
            jobs = {
                "RECENT NEWS": self._safe("news", researcher.invoke(self._news_research_prompt(question))),
                "ASKNEWS HEADLINES": self._safe("asknews", self._asknews_research(question)),
            }
            if self.USE_BASE_RATE_RESEARCH:
                jobs["BASE RATES, RESOLUTION SOURCE & UPCOMING CATALYSTS"] = self._safe(
                    "base rates", researcher.invoke(self._base_rate_research_prompt(question))
                )
            if self.USE_MARKET_ODDS:
                jobs["PREDICTION MARKET PRICES"] = self._safe(
                    "markets", self._market_odds_research(question)
                )
            if self.USE_SERIES_MODEL and isinstance(question, NumericQuestion):
                jobs["STATISTICAL MODEL (historical-volatility simulation)"] = self._safe(
                    "series model", self._series_model_research(question)
                )
            results = await asyncio.gather(*jobs.values())
            sections = [
                f"=== {title} ===\n{text.strip()}"
                for title, text in zip(jobs.keys(), results)
                if text and text.strip()
            ]
            if not sections:
                raise RuntimeError("All research sources failed for this question")
            research = "\n\n".join(sections)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    # ---------- aggregation ----------

    async def _aggregate_predictions(self, predictions, question):
        import math

        if isinstance(question, BinaryQuestion):
            clipped = [min(max(float(p), 0.01), 0.99) for p in predictions]
            mean_log_odds = sum(math.log(p / (1 - p)) for p in clipped) / len(clipped)
            pooled = 1 / (1 + math.exp(-mean_log_odds))
            final = min(max(pooled, self.BINARY_FLOOR), self.BINARY_CEILING)
            logger.info(
                f"Aggregated {len(clipped)} forecasts {[round(p, 3) for p in clipped]} -> {final:.3f}"
            )
            return final
        aggregate = await super()._aggregate_predictions(predictions, question)
        if isinstance(question, MultipleChoiceQuestion):
            from forecasting_tools.data_models.multiple_choice_report import PredictedOption

            floored = [max(o.probability, self.MC_FLOOR) for o in aggregate.predicted_options]
            total = sum(floored)
            aggregate = PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name=o.option_name, probability=p / total)
                    for o, p in zip(aggregate.predicted_options, floored)
                ]
            )
        return aggregate

    ######################### END OWL360 UPGRADES ##########################

    @staticmethod
    def _get_research_prompt(
        question: MetaculusQuestion, researcher: str | GeneralLlm
    ) -> str:
        if GeneralLlm.to_model_name(researcher) == "asknews/news-summaries":
            return question.question_text

        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question.question_text}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}
            """
        )
        return prompt

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        reasoning = await self._invoke_forecaster(question, prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a probability forecast for a binary question.
            {self._create_resolved_question_parsing_message()}
            """
        )
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of a scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            {self._create_resolved_question_parsing_message()}
            """
        )
        reasoning = await self._invoke_forecaster(question, prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._invoke_forecaster(question, prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ. No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        forecast = await self._date_prompt_to_forecast(question, prompt)
        return forecast

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._invoke_forecaster(question, prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_resolved_question_parsing_message(self) -> str:
        return "- If the text concludes that the question has already resolved *in the past* (i.e. it treats the question as decided rather than something to forecast), please DO NOT return a parsed output, even if a final forecast is also given. Instead indicate that the answer is not explicitly given in the text.\n"

    def _create_single_distribution_parsing_message(
        self, question: NumericQuestion | DateQuestion
    ) -> str:
        message = (
            "- The text may contain multiple percentile distributions (e.g. forecasts for several related questions/entities, or intermediate drafts before a final answer). You must return exactly ONE distribution: the single final distribution that answers the question stated above.\n"
            "- Never merge or concatenate percentile lists that refer to different entities, options, or scenarios. Each percentile should appear at most once in your output.\n"
            "- If there are multiple final distributions and you cannot tell which one answers the question stated above, do not guess or combine them. Instead indicate that the answer is not explicitly given in the text."
        )
        if question.group_question_option is not None:
            message += (
                f'\n- This question is specifically about "{question.group_question_option}" (one subquestion within a group of related questions). '
                f'If the text gives distributions for multiple subjects, return only the distribution for "{question.group_question_option}".'
            )
        return message

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the template forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    check_environment(strict=True)
    publish_to_metaculus = True
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)

    # Configure the bot. The `llms=` block below is commented out to use
    # whichever default models forecasting-tools picks based on your env vars;
    # uncomment and edit to pin specific models.
    template_bot = FallTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            "default": GeneralLlm(
                model="openrouter/openai/gpt-5.4",
                timeout=120,
                allowed_tries=4,
            ),
            "summarizer": "openrouter/openai/gpt-5.4-mini",
            "researcher": "openrouter/perplexity/sonar-pro-search",
            "parser": GeneralLlm(
                model="openrouter/openai/gpt-5-mini",
                timeout=60,
                allowed_tries=5,
            ),
        },
    )

    # Per-mode tournament URL shown in the summary banner footer. These
    # piggyback on the forecasting_tools SDK constants and need updating
    # whenever those rotate seasons.
    TOURNAMENT_URLS = {
        "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
        "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    # Dispatch on mode. Each branch produces a list of ForecastReport (or
    # exceptions, since return_exceptions=True) which then flows into the
    # summary printers below.
    client = MetaculusClient()
    # COST CONTROL: we joined the Fall seasonal tournament late, so it isn't worth the API
    # spend. Set this to True for the Spring season to forecast on both again.
    FORECAST_ON_SEASONAL_TOURNAMENT = False
    if run_mode == "tournament":
        seasonal_tournament_reports = (
            asyncio.run(
                template_bot.forecast_on_tournament(
                    client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
                )
            )
            if FORECAST_ON_SEASONAL_TOURNAMENT
            else []
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # The Metaculus Cup may be uninitialized near the start of a season
        # (Jan/May/Sep). MetaculusClient.ACX_2025_TOURNAMENT = 32564 and
        # MetaculusClient.AI_2027_TOURNAMENT_ID = "ai-2027" are also valid
        # targets here.
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # The bot-testing-area tournament contains all question types and is
        # the recommended target for smoke-testing your bot.
        # https://www.metaculus.com/tournament/bot-testing-area/
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                "bot-testing-area", return_exceptions=True
            )
        )

    template_bot.log_report_summary(forecast_reports)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
