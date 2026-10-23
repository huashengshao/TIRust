"""Small display helpers for the TIRust gallery; no reduction or algebra here."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
from math import isfinite, log, log2
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence


KERNEL_SOURCE = "validation/benchmark/baseline-6b24c03-licensed/results.json"
KERNEL_SHA256 = "6469c9c172f32ce3b9dea53defa9d716856c107bccce7669919ce1edc9609793"
KERNEL_REVISION = "6b24c0372022e22606a1ac85f30c7fa579fb41f0"
COLLIER_SOURCE = "validation/collier-reference/timing-initial8/summary.json"
COLLIER_SHA256 = "d7cfc7eae1f14eaad7b67a4bf839be40f2c64a2a4851f5dc8833ce3329235cc3"
JOINT_SOURCE = "validation/selected-native-four-engine/posthoc-revision2/summary.json"
JOINT_SHA256 = "424cb4028459324b796085e5446c979f75de796eb0aa747e6f2335f654116275"
JOINT_QUALIFICATION_SOURCE = "validation/selected-native-four-engine/posthoc-revision2/qualification.json"
JOINT_QUALIFICATION_SHA256 = "d2b079c44198e8dec4a4f9ac4ec587a2a05984fa267c2a6a9e76d5041186cea4"
JOINT_FROZEN_SOURCE = "validation/selected-native-four-engine/v3-frozen/qualification.json"
JOINT_FROZEN_SHA256 = "58f8492040cd4de48f14b5dedfc6eb1707719297f737ec18dcd98cae0fec8035"
COHORT_CHOICES = {
    "Private selected TIRust / COLLIER · posthoc": "selected-collier",
    "Private selected TIRust / IREGI · posthoc": "selected-iregi",
    "Private baseline TIRust / COLLIER · posthoc": "baseline-collier",
    "Private baseline TIRust / IREGI · posthoc": "baseline-iregi",
    "Private selected / baseline TIRust · posthoc": "selected-baseline",
    "Historical production kernel · TIRust / IREGI": "historical",
}


def project_root(start: str | Path) -> Path:
    """Find the checkout containing the notebook's committed data and examples."""
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "PLAN.md").is_file() and (candidate / "Cargo.toml").is_file():
            return candidate
    raise FileNotFoundError("Open this gallery from the TIRust source checkout.")


def theme() -> str:
    return """<style>
    .tirust { --navy:#132e4b; --ink:#233549; --muted:#5b6b7c;
      --border:#dce5ed; --paper:#f7fafc; color:var(--ink); }
    .tirust-hero { border-top:5px solid var(--navy); padding:1.55rem 1.7rem;
      background:linear-gradient(120deg,#f3f7fb,#fff); border-radius:0 0 12px 12px; }
    .tirust-eyebrow { font:600 .72rem system-ui; letter-spacing:.16em;
      text-transform:uppercase; color:#4b6986; margin-bottom:.65rem; }
    .tirust h1 { color:var(--navy); font:400 2.6rem Georgia,serif;
      letter-spacing:-.025em; margin:0 0 .55rem; line-height:1.1; }
    .tirust-subtitle { max-width:56rem; line-height:1.65; color:var(--muted); }
    .tirust-cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(165px,1fr));
      gap:.85rem; margin:.9rem 0; }
    .tirust-card { padding:1rem 1.1rem; border:1px solid var(--border);
      border-radius:10px; background:var(--paper); min-width:0; }
    .tirust-card-label { color:var(--muted); font:.75rem system-ui;
      letter-spacing:.035em; margin-bottom:.55rem; }
    .tirust-card-value { color:var(--navy); font:500 1.05rem ui-monospace,monospace;
      overflow-wrap:anywhere; line-height:1.5; }
    .tirust-table { border-collapse:separate; border-spacing:0 .45rem;
      width:100%; font:.88rem system-ui; }
    .tirust-table th { text-align:left; padding:.6rem .75rem; color:var(--muted);
      font-weight:500; border-bottom:1px solid var(--border); }
    .tirust-table td { background:#f5f8fb; padding:.9rem .75rem; }
    .tirust-table td:first-child { border-radius:8px 0 0 8px; }
    .tirust-table td:last-child { border-radius:0 8px 8px 0; }
    .tirust-ratio { font:600 1.3rem Georgia,serif; min-width:115px; }
    .tirust-small { display:block; font:.69rem system-ui; line-height:1.5;
      font-weight:400; color:var(--muted); margin-top:.25rem; }
    .tirust-overflow { overflow-x:auto; }
    .tirust-chart { width:100%; min-width:930px; height:auto; }
    @media (prefers-color-scheme:dark) {
      .tirust { --navy:#d1e4f7; --ink:#d8e4f1; --muted:#a3b4c6;
        --border:#34495e; --paper:#182b3e; }
      .tirust-hero { background:linear-gradient(120deg,#182b3e,#102131); }
      .tirust-table td { background:#182b3e; }
      .tirust-eyebrow { color:#a3c4e3; }
    }
    </style>"""


def hero(title: str, subtitle: str, *, eyebrow: str = "TIRust · one-loop integrals") -> str:
    return (
        '<section class="tirust tirust-hero">'
        f'<div class="tirust-eyebrow">{escape(eyebrow)}</div>'
        f"<h1>{escape(title)}</h1>"
        f'<div class="tirust-subtitle">{escape(subtitle)}</div></section>'
    )


def cards(items: Iterable[tuple[str, Any]]) -> str:
    """Format values directly as strings, preserving high-precision digits."""
    content = "".join(
        '<div class="tirust-card">'
        f'<div class="tirust-card-label">{escape(label)}</div>'
        f'<div class="tirust-card-value">{escape(str(value))}</div></div>'
        for label, value in items
    )
    return f'<div class="tirust tirust-cards">{content}</div>'


def laurent_cards(values: Sequence[Any], names: Sequence[str]) -> str:
    if len(values) != 4 or len(names) != 4:
        raise ValueError("Expected the four separated native Laurent coefficients.")
    return cards(zip(names, values, strict=True))


def _checked_json(root: Path, relative: str, expected_sha256: str) -> dict[str, Any]:
    payload = (root / relative).read_bytes()
    actual = sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"Benchmark evidence changed: {relative}. "
            "Review the new source and update the gallery's evidence binding."
        )
    return json.loads(payload)


@dataclass(frozen=True)
class BenchmarkRow:
    family: str
    propagators: int
    rank: int
    batch: int
    tirust_us: float | None
    reference_us: float | None
    ratio: float | None
    ci99: tuple[float, float] | None
    numerical_status: str
    timing_status: str
    max_error: float | None
    tirust_label: str = "TIRust"
    reference_label: str = "IREGI"


def load_kernel_snapshot(root: Path) -> tuple[dict[str, Any], tuple[BenchmarkRow, ...]]:
    """Load one paired run. Ratios are TIRust / IREGI exactly as measured."""
    data = _checked_json(root, KERNEL_SOURCE, KERNEL_SHA256)
    rows = []
    for row in data["rows"]:
        width = row["batch_width"]
        ratio = row.get("ratio") or {}
        ours = (row.get("rust") or {}).get("total") or {}
        reference = (row.get("iregi") or {}).get("total") or {}
        ci = ratio.get("ci99")
        correctness = row.get("correctness") or {}
        rows.append(BenchmarkRow(
            family=row["id"], propagators=row["n"], rank=row["rank"], batch=width,
            tirust_us=ours.get("median_ns", 0) / (1000 * width) if ours else None,
            reference_us=reference.get("median_ns", 0) / (1000 * width) if reference else None,
            ratio=ratio.get("median"), ci99=tuple(ci) if ci is not None else None,
            numerical_status=correctness.get("status", row["status"]),
            timing_status=ratio.get("gate", row["status"]),
            max_error=correctness.get("max_error"),
        ))
    return data, tuple(rows)


def load_reference_times(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Original IREGI and COLLIER times only: no synthetic TIRust ratio."""
    data = _checked_json(root, COLLIER_SOURCE, COLLIER_SHA256)
    rows = [{
        "Family": f'N{row["n"]} · rank {row["rank"]}',
        "Batch": row["batch_width"],
        "IREGI (µs/point)": row["iregi_median_us_per_point"],
        "COLLIER (µs/point)": row["collier_median_us_per_point"],
        "Numerics": row["numerical_status"],
        "Timing qualification": (
            "qualified paired timing" if row["performance_status"] == "improvement"
            else row["performance_status"]
        ),
    } for row in data["rows"]]
    return data, rows


def load_joint_snapshot(
    root: Path, numerator: str = "candidateTIRust", denominator: str = "collier",
) -> tuple[dict[str, Any], tuple[BenchmarkRow, ...]]:
    """Read a saved same-quartet comparison; never reconstruct a timing ratio."""
    labels = {
        "baselineTIRust": "TIRust (baseline)",
        "candidateTIRust": "TIRust (selected)",
        "iregi": "IREGI", "collier": "COLLIER",
    }
    allowed = {
        ("baselineTIRust", "iregi"), ("baselineTIRust", "collier"),
        ("candidateTIRust", "iregi"), ("candidateTIRust", "collier"),
        ("candidateTIRust", "baselineTIRust"),
    }
    if (numerator, denominator) not in allowed:
        raise ValueError("Choose a measured comparison with TIRust in the numerator.")
    data = _checked_json(root, JOINT_SOURCE, JOINT_SHA256)
    qualified = _checked_json(root, JOINT_QUALIFICATION_SOURCE, JOINT_QUALIFICATION_SHA256)
    frozen = _checked_json(root, JOINT_FROZEN_SOURCE, JOINT_FROZEN_SHA256)
    if (data["global_admission_errors"] or not qualified["posthoc"]
            or qualified["new_timing_or_kernel_runs"] != 0
            or frozen["numerical_admission"] != "passed-all8-before-and-after-and-controls"):
        raise ValueError("The frozen four-engine qualification does not admit this display.")
    rows = []
    for row in data["rows"]:
        if row["admission_errors"]:
            raise ValueError(f"Unqualified numerical family: {row['id']}")
        measured = next(r for r in row["ratios"]
                        if (r["numerator"], r["denominator"]) == (numerator, denominator))
        stats = measured["statistics"]
        width = row["batch"]
        prefix, rank, _ = row["id"].split("-", 2)
        rows.append(BenchmarkRow(
            family=row["id"], propagators=int(prefix[1:]), rank=int(rank[1:]), batch=width,
            tirust_us=median(row["engine_samples_ns"][numerator]) / (1000 * width),
            reference_us=median(row["engine_samples_ns"][denominator]) / (1000 * width),
            ratio=stats["median"], ci99=tuple(stats["ci99"]),
            numerical_status="passed", timing_status=measured["gate"], max_error=None,
            tirust_label=labels[numerator], reference_label=labels[denominator],
        ))
    return data, tuple(rows)


def load_benchmark_cohort(root: Path, selection: str) -> tuple[dict[str, str], tuple[BenchmarkRow, ...]]:
    """Keep historical production and private posthoc cohorts explicitly separate."""
    if selection == "historical":
        data, rows = load_kernel_snapshot(root)
        return {
            "source": KERNEL_SOURCE, "snapshot": KERNEL_REVISION[:7],
            "protocol": "41 paired blocks · 99% CI",
            "scope": "Historical production kernel: all eight numerical cases pass; seven regressions and one noisy timing remain visible. Later public APIs and private candidates are not timed by this run.",
            "kernel_scope": data["protocol"]["kernel_scope"],
            "cache_policy": data["protocol"]["cache_policy"],
            "analysis": "Original paired analysis; no later cohort or candidate ratio is combined with it.",
        }, rows
    pairs = {
        "selected-collier": ("candidateTIRust", "collier"),
        "selected-iregi": ("candidateTIRust", "iregi"),
        "baseline-collier": ("baselineTIRust", "collier"),
        "baseline-iregi": ("baselineTIRust", "iregi"),
        "selected-baseline": ("candidateTIRust", "baselineTIRust"),
    }
    if selection not in pairs:
        raise ValueError("Choose one of the separately measured benchmark cohorts.")
    _, rows = load_joint_snapshot(root, *pairs[selection])
    return {
        "source": JOINT_SOURCE, "snapshot": "Private v3 · posthoc revision 2",
        "protocol": "48 four-engine quartets · 99% CI",
        "scope": "Historical private prototypes, not current production: baseline uses Symbolica masters; selected uses native A0/B0/C0 for plain f64. No backend adoption or complete performance acceptance follows. Bubble rank 0 singleton remains short-block inconclusive.",
        "kernel_scope": "Complete tensor/Laurent reduction call for identical massive Euclidean N2/N3 ranks 0/1, batches 1/32. Preparation and subprocess transport excluded; every array and pole probe checked before and after timing.",
        "cache_policy": "Reused workspaces, numerical recomputation and cleared reference caches; ten warm calls, 48 quartets covering all 24 engine permutations twice, CPU 254. Every block must reach 100 ms; the one 99.893971 ms block remains inconclusive.",
        "analysis": "Explicit posthoc saved-data correction: original conservative 35 cohort-failed/five short verdicts remain archived. Independently reviewed clean-terminal checks permit family gates while all 40 sample arrays, medians, 99% CIs and widths remain unchanged. No repeated timing, altered threshold or cross-session quotient. " + JOINT_QUALIFICATION_SOURCE,
    }, rows


def _ratio_background(row: BenchmarkRow) -> str:
    if row.ratio is None or not isfinite(row.ratio) or row.ratio <= 0:
        return "#e9edf1"
    if row.timing_status.startswith("inconclusive"):
        return "#fff0c7"
    magnitude = min(abs(log2(row.ratio)) / 2, 1)
    color = (202, 232, 215) if row.ratio <= 1 else (242, 201, 196)
    return "#" + "".join(f"{round(255 + (channel - 255) * magnitude):02x}" for channel in color)


def benchmark_grid(rows: Sequence[BenchmarkRow]) -> str:
    numerator, denominator = _comparison_labels(rows)
    header = "".join(f"<th>{escape(name)}</th>" for name in (
        "Family", "Batch", f"{numerator} µs/point", f"{denominator} µs/point",
        f"{numerator} / {denominator}", "Qualification",
    ))
    body = []
    for row in rows:
        ratio = "—" if row.ratio is None else f"{row.ratio:.3f}"
        interval = "Unmeasured" if row.ci99 is None else f"99% CI [{row.ci99[0]:.3f}, {row.ci99[1]:.3f}]"
        ours = "—" if row.tirust_us is None else f"{row.tirust_us:.3f}"
        reference = "—" if row.reference_us is None else f"{row.reference_us:.3f}"
        body.append(
            f'<tr><td>N{row.propagators} · rank {row.rank}</td><td>{row.batch}</td>'
            f"<td>{ours}</td><td>{reference}</td>"
            f'<td class="tirust-ratio" style="background:{_ratio_background(row)};color:#17324b">'
            f'{ratio}<span class="tirust-small" style="color:#42566a">{interval}</span></td>'
            f'<td>{escape(row.timing_status)}<span class="tirust-small">'
            f'Numerics: {escape(row.numerical_status)}</span></td></tr>'
        )
    return '<div class="tirust tirust-overflow"><table class="tirust-table"><thead><tr>' + header + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"


def ratio_chart(rows: Sequence[BenchmarkRow]) -> str:
    numerator, denominator = _comparison_labels(rows)
    measured = [row for row in rows if row.ratio is not None and row.ci99 is not None]
    if not measured:
        return '<div class="tirust">No paired ratio is measured in this selection.</div>'
    lower = min(.5, *(row.ci99[0] for row in measured)) / 1.15
    upper = max(2, *(row.ci99[1] for row in measured)) * 1.15
    x0, x1, width = 175, 830, 930
    height = 65 + 44 * len(measured)

    def position(value: float) -> float:
        return x0 + (log(value) - log(lower)) / (log(upper) - log(lower)) * (x1 - x0)

    marks = []
    for tick in (.25, .5, 1, 2, 4, 8):
        if lower <= tick <= upper:
            x = position(tick)
            stroke = "#183b5b" if tick == 1 else "#dce5ed"
            marks.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="15" y2="{height-35}" stroke="{stroke}"/>')
            marks.append(f'<text x="{x:.2f}" y="{height-14}" text-anchor="middle" fill="#5b6b7c">{tick:g}</text>')
    for index, row in enumerate(measured):
        y = 33 + 44 * index
        label = f"N{row.propagators} · rank {row.rank} · batch {row.batch}"
        color = "#aa7821" if row.timing_status.startswith("inconclusive") else ("#30734d" if row.ratio < 1 else "#ae4941")
        marks.append(f'<text x="8" y="{y+4}" fill="#42566a">{escape(label)}</text>')
        marks.append(f'<line x1="{position(row.ci99[0]):.2f}" x2="{position(row.ci99[1]):.2f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        marks.append(f'<circle cx="{position(row.ratio):.2f}" cy="{y}" r="5" fill="{color}"/>')
    title = escape(f"{numerator} divided by {denominator}: paired medians with 99 percent confidence intervals; below one is faster")
    return f'<div class="tirust tirust-overflow"><svg class="tirust-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{title}" style="font:12px system-ui;background:#fff;border-radius:10px"><title>{title}</title>' + "".join(marks) + "</svg></div>"


def _comparison_labels(rows: Sequence[BenchmarkRow]) -> tuple[str, str]:
    labels = {(row.tirust_label, row.reference_label) for row in rows}
    if len(labels) > 1:
        raise ValueError("Display separately measured comparisons in separate tables.")
    return next(iter(labels), ("TIRust", "IREGI"))
