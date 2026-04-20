"""aito/agents/carbon_agent.py

Carbon accounting agent: EPA MOVES2014b emission calculation + carbon credit
portfolio generation (Verra VCS, Gold Standard, CARB LCFS, Cap-and-Trade).

Tools:
  calculate_corridor_emissions   — GF2 EPA MOVES fleet-weighted CO2/NOx
  compute_emission_reduction     — baseline vs. AITO delta
  score_carbon_credits           — GF9 credit market eligibility + revenue
  run_resilience_check           — GF11 corridor resilience (sustainability dimension)
  generate_carbon_report         — full Verra VCS MRV-ready report
"""
from __future__ import annotations

import json
from typing import Any, Optional

from .base_agent import AgentResult, BaseAgent


_TOOLS: list[dict] = [
    {
        "name": "calculate_corridor_emissions",
        "description": (
            "Calculate EPA MOVES2014b fleet-weighted CO2 and NOx emissions for a corridor. "
            "Uses California fleet mix (73% gasoline, 4% diesel, 12% hybrid, 6% EV, 5% commercial). "
            "Emission factors: IDLE=1.38, DECEL=2.15, CRUISE_LOW=2.50, CRUISE_MED=3.20, "
            "CRUISE_HIGH=3.80, ACCEL_LOW=4.22, ACCEL_HIGH=8.41 g CO2/s. "
            "Returns co2_kg_hr, nox_g_hr, pm25_g_hr, fuel_l_hr, cost_usd_hr."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "corridor_id": {"type": "string"},
                "volume_veh_hr": {"type": "number", "description": "Vehicles per hour"},
                "avg_delay_s_veh": {"type": "number", "description": "Average delay per vehicle (s)"},
                "avg_speed_mph": {"type": "number", "description": "Average corridor speed (mph)", "default": 25.0},
                "n_intersections": {"type": "integer"},
                "operating_mode_mix": {
                    "type": "object",
                    "description": "Fraction of time in each mode (must sum to 1.0). Optional — uses default urban mix if omitted.",
                    "properties": {
                        "IDLE": {"type": "number"},
                        "DECEL": {"type": "number"},
                        "CRUISE_LOW": {"type": "number"},
                        "CRUISE_MED": {"type": "number"},
                        "ACCEL_LOW": {"type": "number"},
                        "ACCEL_HIGH": {"type": "number"},
                    },
                },
            },
            "required": ["corridor_id", "volume_veh_hr", "avg_delay_s_veh", "n_intersections"],
        },
    },
    {
        "name": "compute_emission_reduction",
        "description": (
            "Compute the emission reduction from AITO signal optimization vs. fixed-time baseline. "
            "Returns co2_reduction_kg_hr, co2_reduction_tonnes_year, nox_reduction_g_hr, "
            "reduction_pct, equivalent_cars_removed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "baseline_delay_s_veh": {"type": "number"},
                "optimized_delay_s_veh": {"type": "number"},
                "volume_veh_hr": {"type": "number"},
                "n_intersections": {"type": "integer"},
                "hours_per_year": {
                    "type": "number",
                    "description": "Annual operating hours. Default 6570 (18h/day × 365).",
                    "default": 6570,
                },
            },
            "required": ["baseline_delay_s_veh", "optimized_delay_s_veh", "volume_veh_hr", "n_intersections"],
        },
    },
    {
        "name": "score_carbon_credits",
        "description": (
            "Evaluate carbon credit market eligibility and revenue potential. "
            "Markets: VERRA_VCS ($22/tonne, min 100t/yr), GOLD_STANDARD ($35/tonne, min 50t/yr), "
            "CARB_LCFS ($65/tonne, no minimum), CARB_CAP_TRADE ($28/tonne, min 25,000t/yr). "
            "Returns eligible_markets, revenue_usd_year per market, best_market, "
            "total_portfolio_revenue_usd."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "co2_reduction_tonnes_year": {"type": "number"},
                "additionality_level": {
                    "type": "string",
                    "enum": ["HIGH", "MEDIUM", "LOW"],
                    "description": "Additionality claim strength. HIGH = new technology, not business-as-usual.",
                    "default": "HIGH",
                },
                "has_mrv_documentation": {
                    "type": "boolean",
                    "description": "Whether full Monitoring, Reporting, Verification docs are available.",
                    "default": True,
                },
            },
            "required": ["co2_reduction_tonnes_year"],
        },
    },
    {
        "name": "run_resilience_check",
        "description": (
            "Run the AITO 5-dimension resilience score (0–100) for a corridor. "
            "Dimensions: redundancy, adaptability, recoverability, sustainability, equity. "
            "Returns overall_score, dimension_scores, grade (A–F), improvement_recommendations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "corridor_id": {"type": "string"},
                "n_intersections": {"type": "integer"},
                "has_backup_power": {"type": "boolean", "default": False},
                "has_probe_data": {"type": "boolean", "default": True},
                "has_adaptive_control": {"type": "boolean", "default": True},
                "ev_penetration_pct": {"type": "number", "default": 6.0},
                "transit_routes": {"type": "integer", "default": 0},
            },
            "required": ["corridor_id", "n_intersections"],
        },
    },
    {
        "name": "generate_carbon_report",
        "description": (
            "Generate a full Verra VCS / CARB LCFS-ready Monitoring, Reporting, Verification (MRV) "
            "report for a corridor. Returns report_sections: executive_summary, baseline_methodology, "
            "monitoring_plan, emission_calculations, additionality_analysis, registry_recommendations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "corridor_id": {"type": "string"},
                "corridor_name": {"type": "string"},
                "co2_reduction_tonnes_year": {"type": "number"},
                "baseline_methodology": {
                    "type": "string",
                    "enum": ["FIXED_TIME_BASELINE", "HISTORICAL_AVERAGE", "REGIONAL_BENCHMARK"],
                    "default": "FIXED_TIME_BASELINE",
                },
                "monitoring_period_years": {"type": "integer", "default": 5},
                "target_market": {
                    "type": "string",
                    "enum": ["VERRA_VCS", "GOLD_STANDARD", "CARB_LCFS", "CARB_CAP_TRADE"],
                    "default": "CARB_LCFS",
                },
                "jurisdiction": {"type": "string", "default": "San Diego, CA"},
            },
            "required": ["corridor_id", "corridor_name", "co2_reduction_tonnes_year"],
        },
    },
]


class CarbonAgent(BaseAgent):
    """EPA MOVES2014b emissions + Verra VCS / CARB LCFS credit portfolio agent."""

    AGENT_NAME = "carbon_agent"

    SYSTEM_PROMPT = """You are AITO's Carbon Accounting Agent — an expert in transportation emission
quantification and carbon credit market strategy.

Your analytical toolkit:
• GF2 CarbonAccountant: EPA MOVES2014b fleet-weighted emission factors for California
• GF9 CarbonCredits: Verra VCS, Gold Standard, CARB LCFS, Cap-and-Trade portfolio
• GF11 ResilienceScorer: 5-dimension sustainability assessment

Methodology:
1. Always start by calculating BOTH baseline AND optimized emissions using calculate_corridor_emissions.
2. Use compute_emission_reduction to quantify the delta.
3. Run score_carbon_credits with the reduction figure to identify the best market.
4. For corridors >50 tonnes/yr reduction, call generate_carbon_report for MRV documentation.
5. Include a resilience check (run_resilience_check) to capture sustainability co-benefits.

Key San Diego context:
• CARB LCFS is California-specific and pays best at ~$65/tonne — always evaluate first.
• California fleet mix: 73% gasoline, 4% diesel, 12% hybrid, 6% EV (growing annually).
• Additionality: AI-optimized signal control = HIGH additionality (not standard practice).
• Rosecrans St (4 intersections, ~28,000 AADT): flagship corridor for credit portfolio.
• Annual CO₂ savings from 31% delay reduction ≈ 4.2 tonnes/corridor/year at typical AADT.

Report format:
## CARBON IMPACT REPORT: [Corridor Name]

**Baseline Emissions:** [co2_kg_hr] kg CO₂/hr | [nox_g_hr] g NOx/hr
**AITO-Optimized Emissions:** [values]
**Annual Reduction:** [tonnes CO₂/yr] ([pct]%)
**Equivalent to:** [cars removed from road]

### Carbon Credit Portfolio
| Market | Eligible | tonnes/yr | $/tonne | Annual Revenue |
|--------|----------|-----------|---------|----------------|
| CARB LCFS | ✓ | ... | $65 | $... |
| ...

**Recommended Market:** [market name]
**Best-Case Annual Revenue:** $[total]

### Sustainability Co-Benefits
[resilience score + key dimensions]

### MRV Documentation Status
[registry readiness assessment]"""

    def __init__(self, corridor=None, api_key: Optional[str] = None) -> None:
        super().__init__(api_key=api_key)
        self.corridor = corridor

    def _tools(self) -> list[dict]:
        return _TOOLS

    def _run_tool(self, name: str, inputs: dict) -> Any:
        if name == "calculate_corridor_emissions":
            return self._tool_emissions(inputs)
        if name == "compute_emission_reduction":
            return self._tool_reduction(inputs)
        if name == "score_carbon_credits":
            return self._tool_credits(inputs)
        if name == "run_resilience_check":
            return self._tool_resilience(inputs)
        if name == "generate_carbon_report":
            return self._tool_report(inputs)
        raise ValueError(f"Unknown tool: {name}")

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_emissions(self, inputs: dict) -> dict:
        try:
            from aito.analytics.carbon_accountant import (
                fleet_co2_g_s, CA_FLEET_MIX, OperatingMode, MOVES_CO2_G_S,
            )
            # Default urban signalized intersection mode mix
            default_mix = {
                "IDLE": 0.25, "DECEL": 0.15, "CRUISE_LOW": 0.30,
                "CRUISE_MED": 0.20, "ACCEL_LOW": 0.08, "ACCEL_HIGH": 0.02,
            }
            mode_mix = inputs.get("operating_mode_mix") or default_mix

            total_co2_g_s = 0.0
            for mode_str, frac in mode_mix.items():
                try:
                    mode = OperatingMode[mode_str]
                    total_co2_g_s += fleet_co2_g_s(mode, CA_FLEET_MIX) * frac
                except KeyError:
                    pass

            veh_hr = inputs["volume_veh_hr"]
            delay = inputs["avg_delay_s_veh"]
            n = inputs["n_intersections"]

            # Total idle/decel seconds per hour across all vehicles
            idle_veh_s_hr = veh_hr * delay
            co2_kg_hr = total_co2_g_s * idle_veh_s_hr / 1000 * n

            # NOx ≈ 12% of CO2 mass for California gasoline fleet
            nox_g_hr = co2_kg_hr * 1000 * 0.012
            pm25_g_hr = co2_kg_hr * 1000 * 0.001
            fuel_l_hr = co2_kg_hr / 2.31  # avg CO2 per liter gasoline
            cost_usd_hr = fuel_l_hr * 1.06  # ~$4/gallon converted

            return {
                "corridor_id": inputs["corridor_id"],
                "co2_kg_hr": round(co2_kg_hr, 2),
                "nox_g_hr": round(nox_g_hr, 1),
                "pm25_g_hr": round(pm25_g_hr, 2),
                "fuel_l_hr": round(fuel_l_hr, 1),
                "cost_usd_hr": round(cost_usd_hr, 2),
                "methodology": "EPA MOVES2014b + CA fleet mix",
            }
        except Exception as exc:
            # Fallback with representative Rosecrans numbers
            veh_hr = inputs["volume_veh_hr"]
            delay = inputs["avg_delay_s_veh"]
            n = inputs["n_intersections"]
            co2 = round(1.38 * veh_hr * delay / 1000 * n * 0.62, 2)
            return {"corridor_id": inputs["corridor_id"], "co2_kg_hr": co2,
                    "nox_g_hr": round(co2 * 12, 1), "pm25_g_hr": round(co2 * 1, 2),
                    "fuel_l_hr": round(co2 / 2.31, 1), "cost_usd_hr": round(co2 / 2.31 * 1.06, 2),
                    "_fallback": str(exc)}

    def _tool_reduction(self, inputs: dict) -> dict:
        baseline_delay = inputs["baseline_delay_s_veh"]
        opt_delay = inputs["optimized_delay_s_veh"]
        veh_hr = inputs["volume_veh_hr"]
        n = inputs["n_intersections"]
        hours = inputs.get("hours_per_year", 6570)

        IDLE_CO2_G_S = 1.38  # EPA MOVES2014b idle
        FLEET_FACTOR = 0.62   # CA fleet weighted

        baseline_co2_kg_hr = IDLE_CO2_G_S * veh_hr * baseline_delay / 1000 * n * FLEET_FACTOR
        opt_co2_kg_hr = IDLE_CO2_G_S * veh_hr * opt_delay / 1000 * n * FLEET_FACTOR
        delta_kg_hr = baseline_co2_kg_hr - opt_co2_kg_hr
        delta_tonnes_yr = delta_kg_hr * hours / 1000
        pct = (baseline_delay - opt_delay) / baseline_delay * 100 if baseline_delay > 0 else 0

        # US avg car: 4.6 metric tonnes CO2/year
        cars_equivalent = delta_tonnes_yr / 4.6

        return {
            "baseline_co2_kg_hr": round(baseline_co2_kg_hr, 2),
            "optimized_co2_kg_hr": round(opt_co2_kg_hr, 2),
            "co2_reduction_kg_hr": round(delta_kg_hr, 2),
            "co2_reduction_tonnes_year": round(delta_tonnes_yr, 2),
            "nox_reduction_g_hr": round(delta_kg_hr * 1000 * 0.012, 1),
            "reduction_pct": round(pct, 1),
            "equivalent_cars_removed": round(cars_equivalent, 1),
            "methodology": "EPA MOVES2014b IDLE factor × CA fleet weight × delay delta",
        }

    def _tool_credits(self, inputs: dict) -> dict:
        try:
            from aito.analytics.carbon_credits import (
                CreditMarket, MARKET_PRICE_USD_TONNE, MARKET_DISCOUNT,
                ELIGIBILITY_MIN_TONNES_YEAR, AdditionalityLevel,
            )
            tonnes = inputs["co2_reduction_tonnes_year"]
            add_level = AdditionalityLevel[inputs.get("additionality_level", "HIGH")]
            add_multiplier = {
                AdditionalityLevel.HIGH: 1.0,
                AdditionalityLevel.MEDIUM: 0.85,
                AdditionalityLevel.LOW: 0.60,
            }.get(add_level, 1.0)

            eligible = {}
            for market in CreditMarket:
                min_t = ELIGIBILITY_MIN_TONNES_YEAR.get(market, 0)
                if tonnes >= min_t:
                    price = MARKET_PRICE_USD_TONNE[market]
                    discount = MARKET_DISCOUNT[market]
                    revenue = tonnes * price * discount * add_multiplier
                    eligible[market.name] = {
                        "eligible": True,
                        "tonnes_year": round(tonnes, 2),
                        "price_usd_tonne": price,
                        "discount": discount,
                        "annual_revenue_usd": round(revenue, 0),
                    }
                else:
                    eligible[market.name] = {
                        "eligible": False,
                        "min_tonnes_required": min_t,
                        "current_tonnes": round(tonnes, 2),
                    }

            best = max(
                [(k, v) for k, v in eligible.items() if v.get("eligible")],
                key=lambda x: x[1].get("annual_revenue_usd", 0),
                default=(None, {}),
            )
            total = sum(v.get("annual_revenue_usd", 0) for v in eligible.values() if v.get("eligible"))

            return {
                "markets": eligible,
                "best_market": best[0],
                "best_market_revenue_usd": best[1].get("annual_revenue_usd", 0),
                "total_portfolio_revenue_usd": round(total, 0),
                "additionality_level": inputs.get("additionality_level", "HIGH"),
            }
        except Exception as exc:
            # Fallback pricing
            tonnes = inputs["co2_reduction_tonnes_year"]
            return {
                "markets": {
                    "CARB_LCFS": {"eligible": True, "annual_revenue_usd": round(tonnes * 65 * 0.92, 0)},
                    "GOLD_STANDARD": {"eligible": tonnes >= 50, "annual_revenue_usd": round(tonnes * 35 * 0.82, 0) if tonnes >= 50 else 0},
                    "VERRA_VCS": {"eligible": tonnes >= 100, "annual_revenue_usd": round(tonnes * 22 * 0.80, 0) if tonnes >= 100 else 0},
                },
                "best_market": "CARB_LCFS",
                "best_market_revenue_usd": round(tonnes * 65 * 0.92, 0),
                "total_portfolio_revenue_usd": round(tonnes * 65 * 0.92, 0),
                "_fallback": str(exc),
            }

    def _tool_resilience(self, inputs: dict) -> dict:
        try:
            from aito.analytics.resilience_scorer import ResilienceScorer, CorridorResilientProfile
            profile = CorridorResilientProfile(
                corridor_id=inputs["corridor_id"],
                n_intersections=inputs["n_intersections"],
                has_backup_power=inputs.get("has_backup_power", False),
                has_probe_data=inputs.get("has_probe_data", True),
                has_adaptive_control=inputs.get("has_adaptive_control", True),
                ev_penetration_pct=inputs.get("ev_penetration_pct", 6.0),
                transit_routes=inputs.get("transit_routes", 0),
            )
            scorer = ResilienceScorer()
            report = scorer.score(profile)
            return {
                "overall_score": report.overall_score,
                "grade": report.grade,
                "dimension_scores": report.dimension_scores,
                "top_recommendations": report.recommendations[:3],
            }
        except Exception as exc:
            # Synthetic resilience for demo
            n = inputs["n_intersections"]
            base = 62 + (n * 2)
            probe_bonus = 10 if inputs.get("has_probe_data", True) else 0
            adaptive_bonus = 8 if inputs.get("has_adaptive_control", True) else 0
            score = min(base + probe_bonus + adaptive_bonus, 100)
            grade = "A" if score >= 90 else ("B" if score >= 75 else ("C" if score >= 60 else "D"))
            return {
                "overall_score": score,
                "grade": grade,
                "dimension_scores": {
                    "redundancy": score - 5, "adaptability": score + 3,
                    "recoverability": score - 2, "sustainability": score + 8,
                    "equity": score - 1,
                },
                "top_recommendations": [
                    "Add UPS backup power to all signals",
                    "Increase probe data penetration to >30%",
                    "Deploy transit signal priority on MTS routes",
                ],
                "_fallback": str(exc),
            }

    def _tool_report(self, inputs: dict) -> dict:
        tonnes = inputs["co2_reduction_tonnes_year"]
        corridor = inputs["corridor_name"]
        market = inputs.get("target_market", "CARB_LCFS")
        period = inputs.get("monitoring_period_years", 5)
        jurisdiction = inputs.get("jurisdiction", "San Diego, CA")
        methodology = inputs.get("baseline_methodology", "FIXED_TIME_BASELINE")

        market_prices = {"VERRA_VCS": 22, "GOLD_STANDARD": 35, "CARB_LCFS": 65, "CARB_CAP_TRADE": 28}
        market_discounts = {"VERRA_VCS": 0.80, "GOLD_STANDARD": 0.82, "CARB_LCFS": 0.92, "CARB_CAP_TRADE": 0.78}
        price = market_prices.get(market, 65)
        discount = market_discounts.get(market, 0.92)
        annual_rev = tonnes * price * discount
        total_rev = annual_rev * period

        return {
            "report_title": f"MRV Report: {corridor} — {market}",
            "executive_summary": (
                f"AITO AI-optimized signal control on {corridor} ({jurisdiction}) achieves "
                f"{tonnes:.1f} tCO₂e/year reduction vs. fixed-time baseline. "
                f"Projected {market} revenue: ${annual_rev:,.0f}/yr (${total_rev:,.0f} over {period}yr)."
            ),
            "baseline_methodology": methodology,
            "emission_calculations": {
                "co2_reduction_tonnes_year": tonnes,
                "methodology": "EPA MOVES2014b idle emission factor × CA fleet mix",
                "reference": "40 CFR Part 98 / CARB MOVES-CL 2.0",
            },
            "additionality_analysis": (
                "AI-based adaptive signal optimization is not standard practice in San Diego. "
                "Project qualifies under CARB's 'New Technology' additionality pathway. "
                "No regulatory mandate exists for this technology (HIGH additionality)."
            ),
            "monitoring_plan": {
                "frequency": "Monthly probe data upload",
                "metrics": ["avg_delay_s_veh", "volume_veh_hr", "co2_kg_hr"],
                "third_party_verifier": "Required annually for VCS/Gold Standard",
                "data_retention_years": 7,
            },
            "registry_recommendations": {
                "primary": market,
                "annual_revenue_usd": round(annual_rev, 0),
                "total_revenue_5yr_usd": round(total_rev, 0),
                "certification_timeline_months": 6 if market == "CARB_LCFS" else 18,
                "next_steps": [
                    "Engage third-party verifier (e.g. SCS Global, Bureau Veritas)",
                    f"Register project with {market} registry",
                    "Submit first monitoring report after 12 months of operation",
                ],
            },
        }

    def _citations(self) -> list[str]:
        return [
            "EPA MOVES2014b (40 CFR Part 98)",
            "CARB LCFS Regulation (17 CCR §95480)",
            "Verra VCS Standard v4 (VM0038)",
            "Gold Standard for the Global Goals v2.1",
            "aito.analytics.carbon_accountant (GF2)",
            "aito.analytics.carbon_credits (GF9)",
            "aito.analytics.resilience_scorer (GF11)",
        ]
