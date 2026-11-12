#!/usr/bin/env python3
"""Integral families of the quarkonium processes, for amplitude.py.

A family is a parent topology of the largest N (the boxes of gg_QQ, the
pentagons of gg_QQg, each with its propagator mass pattern) together with all
its pinches down to bubbles, every one in the parent's loop-momentum routing:
a member takes the parent's shifts and masses at its propagator subset, as an
amplitude's integrals do (TIRust's AmplitudeWorkspace, Table 9 of the
benchmark report). The rank of each member is its N, the largest of a
renormalizable gauge theory.

The output has the format of points.json, with the member integrals as
topologies (each with "members", the parent's "groups" and "masses", and
"parent"), and a "families" list per process. run.py, affine_reference.py and
coulomb_check.py read it unchanged.

usage: families.py POINTS_JSON OUTPUT_JSON
"""
import itertools
import json
import sys


def main():
    data = json.load(open(sys.argv[1]))
    out = {key: value for key, value in data.items() if key != "processes"}
    out["processes"] = []
    for process in data["processes"]:
        top_n = max(t["n"] for t in process["topologies"])
        members, families = [], []
        for parent in (t for t in process["topologies"] if t["n"] == top_n):
            family = {"parent": parent["id"], "members": []}
            for size in range(top_n, 1, -1):
                for subset in itertools.combinations(range(top_n), size):
                    member = {"id": f"{parent['id']}[{','.join(map(str, subset))}]", "n": size, "rank": size,
                              "groups": parent["groups"], "masses": parent["masses"], "members": list(subset),
                              "parent": parent["id"],
                              "member_masses": [parent["masses"][i] for i in subset]}
                    members.append(member)
                    family["members"].append(member["id"])
            families.append(family)
        out["processes"].append(dict(process, topologies=members, families=families))
        print(f"{process['id']}: {len(families)} families of {len(families[0]['members'])} integrals, "
              f"{len(members)} member integrals, {len(process['points'])} points")
    json.dump(out, open(sys.argv[2], "w"))


if __name__ == "__main__":
    main()
