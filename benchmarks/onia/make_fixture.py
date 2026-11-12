#!/usr/bin/env python3
"""Write the fixture of crates/iregi-rust/tests/onia.rs from a benchmarks/onia run.

Selected lanes of both processes, with their exact binary64 inputs and:
- degenerate (affinely dependent shifts, K != 0): the reference of
  affine_reference.py (exact identity over non-degenerate integrals confirmed
  by quad IREGI or arbitrated by COLLIER), 25 significant digits;
- Coulomb exchange (linearly dependent propagators): no TIRust reference;
  `iregi_reference` holds quad original IREGI's IBP array, which
  coulomb_check.py found equal to the exact doubled-gluon combination of
  IREGI's general API (kept for when TIRust supports these integrals).

usage: make_fixture.py POINTS_JSON AFFINE_DIR COULOMB_JSON OUTPUT_JSON
"""
import json
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402

DEGENERATE = [
    ("gg_QQ", "N4-HLLH-Q|g2|Q|g1", [0, 50]),
    ("gg_QQ", "N4-LHHL-Q|g2|Q|g1", [0, 50]),
    ("gg_QQ", "N4-LLHL-g2|Q|Q|g1", [0, 50]),
    ("gg_QQ", "N3-LHL-Q|Q|g1+g2", [0]),
    ("gg_QQg", "N5-LLLHL-g2|g3|Q|Q|g1", [0]),
    ("gg_QQg", "N5-HLLHH-Q|g2|Q|g3|g1", [1]),
    ("gg_QQg", "N5-LHLLL-Q|Q|g2|g3|g1", [101]),
    ("gg_QQg", "N4-HLLH-Q|g2+g3|Q|g1", [0]),
    ("gg_QQg", "N4-LHHL-Q|g3|Q|g1+g2", [100]),
]
COULOMB = [
    ("gg_QQ", "N4-HHLH-g2|Q|Q|g1", [0]),
    ("gg_QQ", "N3-HLH-Q|Q|g1+g2", [0]),
    ("gg_QQg", "N4-HHLH-g2|Q|Q|g1+g3", [0]),
]


def digits(moments, significant=20):
    return [[[format(Decimal(str(x).strip()), f".{significant - 1}e") for x in pair] for pair in slots]
            for slots in moments]


def main():
    points = json.loads(Path(sys.argv[1]).read_text())
    affine_dir, coulomb = Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
    mu = float(points["mu"])
    affine = {}
    for process in points["processes"]:
        path = affine_dir / f"affine-{process['id']}.jsonl"
        for line in path.read_text().splitlines():
            r = json.loads(line)
            affine[(process["id"], r["topology"], r["index"])] = r
    consistent = {(r["process"], r["topology"], r["index"]): r for r in coulomb}
    cases = []
    for selection, kind in ((DEGENERATE, "degenerate"), (COULOMB, "coulomb")):
        for process_id, topology_id, indices in selection:
            process = next(p for p in points["processes"] if p["id"] == process_id)
            topology = next(t for t in process["topologies"] if t["id"] == topology_id)
            for index in indices:
                point = next(p for p in process["points"] if p["index"] == index)
                lane = run.lanes_of(None, topology, [point])[0]
                case = {"process": process_id, "topology": topology_id, "index": index, "frame": point["frame"],
                        "n": topology["n"], "rank": topology["rank"],
                        "shifts": [[repr(x) for x in q] for q in lane["shifts"]],
                        "masses_squared": [repr(m) for m in lane["masses_squared"]], "kind": kind}
                if kind == "degenerate":
                    record = affine[(process_id, topology_id, index)]
                    assert record["status"] == "ok", record["status"]
                    case["reference"] = digits(record["moments"])
                    case["reference_source"] = ("exact denominator identity over non-degenerate integrals: "
                                                + ", ".join(record["sub_confirmations"]))
                else:
                    check = consistent[(process_id, topology_id, index)]
                    assert check["vs_dotted_combination"]["quad-ibp"] <= 1e-12, check
                    q = run.quad([lane], topology, mu, 0)[0]
                    case["iregi_reference"] = digits(q)
                    case["iregi_reference_source"] = (
                        "quad original IREGI, IBP mode; equal to the exact doubled-gluon combination of IREGI's "
                        f"general API to {check['vs_dotted_combination']['quad-ibp']:.1e}")
                cases.append(case)
    out = {"description": "S-wave quarkonium one-loop integrals (p_c = p_c~), exact lattice inputs; "
                          "written by benchmarks/onia/make_fixture.py",
           "mu": repr(mu), "convention": "IregiNative", "slots": ["uv", "ir1", "ir2", "finite"], "cases": cases}
    Path(sys.argv[4]).write_text(json.dumps(out, indent=None, separators=(",", ":")) + "\n")
    print(f"{len(cases)} cases, {Path(sys.argv[4]).stat().st_size} bytes")


if __name__ == "__main__":
    main()
