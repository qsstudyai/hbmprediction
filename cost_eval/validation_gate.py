"""Evaluate the documented P0/P1 HBM acceptance gates.

This module deliberately treats calibration replay and production validation
as different scopes.  Missing blind/OOM/paired evidence is ``unavailable``;
it can never be silently converted into a passing gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str
    observed: Any
    requirement: str
    scope: str


def _metric_gate(
    name: str,
    values: Mapping[str, Any],
    mean_limit: float,
    worst_limit: float,
) -> GateResult:
    observed = {
        "mean_ape_pct": values.get("mean_ape_pct"),
        "worst_ape_pct": values.get("max_ape_pct"),
    }
    available = all(value is not None for value in observed.values())
    passed = available and (
        observed["mean_ape_pct"] <= mean_limit
        and observed["worst_ape_pct"] <= worst_limit
    )
    return GateResult(
        name=name,
        status="pass" if passed else ("fail" if available else "unavailable"),
        observed=observed,
        requirement=f"mean <= {mean_limit:g}%, worst <= {worst_limit:g}%",
        scope="historical_calibration_replay",
    )


def assess_hbm_gates(
    offline: Mapping[str, Any], calibrated: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Return a non-optimistic audit of the P0/P1 acceptance gates."""
    offline_cases = offline.get("cases", ())
    calibrated_cases = calibrated.get("cases", ())
    offline_labels = {item.get("label_sha256") for item in offline_cases}
    calibrated_labels = {item.get("label_sha256") for item in calibrated_cases}
    if offline_labels != calibrated_labels:
        raise ValueError("offline and calibrated reports use different labels")

    offline_overall = offline["summary"]["overall"]
    calibrated_overall = calibrated["summary"]["overall"]
    covered = sum(
        item["prediction"]["safe_upper_bytes"]
        >= item["measured"]["total_bytes"]
        for item in calibrated_cases
    )
    coverage = covered / len(calibrated_cases) if calibrated_cases else None

    gates = [
        _metric_gate("p0_physical_active", offline_overall["active"], 25, 35),
        _metric_gate("p1_dynamic_point", calibrated_overall["dynamic"], 20, 30),
        _metric_gate("p1_total_point", calibrated_overall["total"], 15, 20),
        GateResult(
            name="p1_safe_upper_coverage",
            status=(
                "pass" if coverage is not None and coverage >= 0.80
                else ("fail" if coverage is not None else "unavailable")
            ),
            observed={
                "covered": covered,
                "total": len(calibrated_cases),
                "coverage": coverage,
            },
            requirement="coverage >= 80%",
            scope="historical_calibration_replay",
        ),
        GateResult(
            "blind_holdout", "unavailable", None,
            "new frozen holdouts for every supported family/layout",
            "production_validation",
        ),
        GateResult(
            "actual_oom_false_safe", "unavailable", None,
            "at least one actual OOM and zero false-safe predictions",
            "production_validation",
        ),
        GateResult(
            "stage_rank_peak_order", "unavailable", None,
            "peak ordering accuracy >= 80% on new labels",
            "production_validation",
        ),
        GateResult(
            "strategy_paired_delta", "unavailable", None,
            "paired strategy delta error <= 20%",
            "production_validation",
        ),
    ]
    serialized = [asdict(item) for item in gates]
    p0_numeric = next(item for item in gates if item.name == "p0_physical_active")
    p1_numeric = [item for item in gates if item.name.startswith("p1_")]
    return {
        "schema": "hbmprediction.validation-gates.v1",
        "evidence_scope": "historical_calibration_replay",
        "label_count": len(calibrated_cases),
        "p0_numeric_status": p0_numeric.status,
        "p1_numeric_status": (
            "pass" if all(item.status == "pass" for item in p1_numeric)
            else "fail"
        ),
        "production_status": "validation_unavailable",
        "gates": serialized,
    }


def render_gate_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# P0/P1 HBM validation gates",
        "",
        f"Evidence scope: `{result['evidence_scope']}`; "
        f"unique labels: {result['label_count']}.",
        "",
        f"P0 numeric: **{result['p0_numeric_status']}**; "
        f"P1 numeric: **{result['p1_numeric_status']}**; "
        f"production: **{result['production_status']}**.",
        "",
        "| Gate | Status | Observed | Requirement | Scope |",
        "|---|---|---|---|---|",
    ]
    for gate in result["gates"]:
        observed = gate["observed"]
        if isinstance(observed, Mapping):
            observed_text = ", ".join(
                f"{key}={value:.3%}" if key == "coverage" and value is not None
                else f"{key}={value:.3f}" if isinstance(value, float)
                else f"{key}={value}"
                for key, value in observed.items()
            )
        else:
            observed_text = "—" if observed is None else str(observed)
        lines.append(
            f"| {gate['name']} | {gate['status']} | {observed_text} | "
            f"{gate['requirement']} | {gate['scope']} |"
        )
    lines.extend((
        "",
        "`unavailable` is not a pass. Historical calibration replay cannot "
        "establish blind generalization, OOM safety, or paired-strategy accuracy.",
        "",
    ))
    return "\n".join(lines)
