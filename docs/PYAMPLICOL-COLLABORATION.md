# pyAmpliCol collaboration protocol (binding)

Received from Valentin Hirschi (pyAmpliCol) and passed to TIRust's agent by
Huasheng Shao on 6 November 2026. It binds every agent working in this
repository and overrides anything conflicting in AGENTS.md, PLAN.md or
HANDOFF.md. The text below is verbatim; change it only by a pull request agreed
with pyAmpliCol. Huasheng answers the freeze window and the licence question
(section 2 item 5) separately.

The repository was renamed from huashengshao/iregi-rust to huashengshao/TIRust
on 5 November 2026. GitHub redirects the old name: the `gh` commands below
(`--repo huashengshao/iregi-rust`), including the freeze check, worked under
both names on 6 November 2026.

```text
You work in github.com/huashengshao/iregi-rust (TIRust). These rules apply from now on and override anything conflicting in AGENTS.md, PLAN.md or HANDOFF.md.

0. YOU NO LONGER OWN main (read this first)
- As agreed with Huasheng, you do not own main any more: pyAmpliCol's coordinator maintains it, and only pyAmpliCol's agents push to it.
- Never push to main, never commit on main and never merge into main yourself: no git push to main, no gh pr merge, no merge button, and never merge your own pull requests.
- Work only on feature branches named hs/<topic>, and propose every change as a pull request against main:
    git fetch origin && git switch -c hs/<topic> origin/main          # new work
    git push -u origin hs/<topic>
    gh pr create --repo huashengshao/iregi-rust --base main --head hs/<topic> --reviewer ValentinHirschi
  Our coordinator reviews and merges it with a merge commit. Never force-push any branch.
- Move your current work in progress off main now: git switch -c hs/<topic> (from your current HEAD), git push -u origin hs/<topic>, then point local main back at origin/main (git branch -f main origin/main, run from the feature branch) and open the pull request. Leave raw captures, logs and PDFs uncommitted (section 5).

1. CONTEXT
- TIRust becomes a tensor-integral reduction backend of pyAmpliCol (github.com/MadGraphTeam/pyamplicol). Huasheng Shao is a co-author.
- Valentin Hirschi's account ValentinHirschi has write access. From now on pyAmpliCol's agents push directly to main; you propose changes by pull request (section 0).

2. WHAT WE CHANGE (our agents do these steps; do not do them yourself, and do not revert any of it)
1) oneloopmaster: the sections of dependencies/oneloopmaster/PATCHES.md are merged into alphal00p/oneloopmaster main (open PRs #17/#18 included); iregi-only code moves here. The dependency becomes oneloop = { git = "https://github.com/alphal00p/oneloopmaster.git", rev = "<merged commit>" }; today's shared base is 128ea6645f2811e6179068b98731501292841409.
2) Symbolica: port to https://github.com/ValentinHirschi/symbolica.git rev 20a9da94071d7dd64ea5cdd6ed9d6e8f8cefeacc (= symbolica-dev/symbolica PR #163; symbolica, numerica and graphica from that rev). Crates require symbolica = "3.0"; the workspace [patch.crates-io] points at that rev. The Symbolica licence forbids distributing a modified Symbolica, so needed Symbolica patches go upstream through our Symbolica PR line, never into this repository.
3) Toolchain: rust-version <= 1.97.1; lib, tests and examples build and pass on Rust 1.97.1; shell.nix (today Rust 1.99) moves to 1.97.1, or a rust-toolchain.toml pins 1.97.1.
4) Hygiene: cargo test --no-run green; tests write only to temporary directories.
5) Licence (after Huasheng's OK): a 0BSD LICENSE for the project's own code and license = "0BSD" in every first-party crate (crates/*); third-party content (dependencies/feynkit-integrals, dependencies/linnest, the fonts under python/tirust/data/render/fonts, python/tirust/data/licenses) keeps its own licence and notices, listed in the LICENSE file.
6) Slimming: validation/ shrinks to the fixtures tests read plus a short summary; outdated docs and data (docs/data/results.json, raw captures, old PDFs) are removed. Before our slimming commit we announce it in an issue: copy (do not move or delete) the validation/ and docs/data/ material you want to keep public into huashengshao/iregi-rust-evidence, without any Symbolica copy, and reply on that issue; we remove it from this repository only after that.
7) Lean core: iregi-rust and iregi-masters work as a git dependency without iregi-prepare, iregi-cli, iregi-python or gammaloop in their graph; the generated evaluators (d0.rs, c0.rs) behind a cargo feature if the binary64 direct engine does not need them.
8) One history rewrite (git filter-repo), after a full mirror backup kept permanently: the large blobs (every docs/data/results.json version, validation captures) and dependencies/symbolica, dependencies/symbolica-patches, dependencies/oneloopmaster (and dependencies/linnest* if de-vendored) leave every commit. See section 4.
9) Later, as normal commits: DD/AP input with exact on-shell legs (supplied S-matrix/kinematics), mu^2 input, an embedder mode (rotation test off, precision tiers and partial DD rescue optional, never the symbolic fallback), precision-relative thresholds (replacing the fixed 2^-40 lightlike-bubble rule), lifting mechanical limits (an N = 7 qualification if needed), a persistent tensor-integral store/provider API with a masters provider, a rotation copy, direct-engine concurrency tests, and possibly kernel speed work (point-batched or SIMD moments, a finite-slot-only mode).

3. COLLABORATION PROTOCOL
- Except during a freeze (section 4): before starting a branch, before committing on top of others' work and before every push, run git fetch origin. Then a branch never pushed: git rebase origin/main. A branch already pushed (even with new local commits): git merge origin/main, never a rebase.
- Small commits, one topic per PR; each PR states what, why, and test results.
- Never reintroduce vendored or modified Symbolica/oneloopmaster (dependencies/symbolica*, dependencies/oneloopmaster), path [patch] entries for shared dependencies, removed large files, or any path stripped by the rewrite (listed on the FREEZE issue).
- Keep stable what pyAmpliCol uses: crates iregi-rust and iregi-masters; DirectReductionPlan; DirectWorkspace<T> for f64, DoubleFloat and Float with its enable_* switches; Convention::OneLoop; TensorWeights::Components; InvariantInput; the batch row layout; moment slots [UV1, IR1, IR2, finite]; BatchResult; LaneStatus; the capability limits in direct/plan.rs; TensorReductionPlan/TensorWorkspace; AmplitudeWorkspace; the conventions (shifts time-first, metric (+,-,-,-), D_i = (q+p_i)^2 - m_i^2, Im m^2 <= 0); every API we add. To change any of them, first open an issue labelled api-change and wait for agreement.
- Tests before every push: until our toolchain and hygiene commits (section 2 items 3-4) are on main, run your current suite (default members) on your current toolchain and your validation checks, and note known failures in the PR. From then on, run cargo +1.97.1 test -p iregi-rust -p iregi-masters plus your default-members suite and your validation checks. New failures block the push; failures that already exist on origin/main do not: report them in an issue.
- Shared pins and the features enabled on Symbolica, numerica, graphica and oneloop (and the Rust toolchain) must equal pyAmpliCol's; change any of them only after an agreed issue labelled pins.

4. FREEZE PROTOCOL (the history rewrite)
- An announcement issue at least 48 h ahead gives the window; it does not freeze. Before the window, push your work in progress to hs/* branches so the rewrite carries it.
- The freeze is an open issue titled exactly "FREEZE: history rewrite"; only that exact title freezes. gh must be installed and authenticated (gh auth status). Before EVERY push of any branch or tag (the section 5 hook also runs it):
    gh issue list --repo huashengshao/iregi-rust --state open --limit 1000 --json title --jq '.[] | select(.title == "FREEZE: history rewrite") | .title'
  Non-empty output, or a failing command, means freeze: push nothing, open no PR, create no tag or release, keep working locally.
- While the FREEZE issue is open, and in the old clone forever after, never run git fetch, pull, rebase or merge against origin; only the new clone talks to origin.
- The closing comment of the FREEZE issue says whether the rewrite happened. If it did not, keep your clone and carry on. If it did:
  - In the old clone, before anything else: git remote set-url --push origin DISABLED-old-history (pushes from it now fail); then rename the directory to iregi-rust.OLD-HISTORY. Never push from the old clone or any branch based on the old history: it would bring the removed blobs back.
  - Re-clone next to the old clone and run git config core.hooksPath .githooks in it. Branches pushed before the freeze are already rewritten on the remote (the issue lists any deleted).
  - Re-apply unpushed local commits as patches (git format-patch / git am), with the commands posted in the issue; default, per branch:
      old clone (never fetched): git format-patch -o ../patches-<topic> ^origin/main ^origin/hs/<topic> hs/<topic>
        (drop ^origin/hs/<topic> if the branch was never pushed)
      new clone: git switch hs/<topic> (or, if absent or deleted, git switch -c hs/<topic> origin/main); if ../patches-<topic> is non-empty: git am -3 ../patches-<topic>/*.patch
  - Old hashes are gone: update provenance (HANDOFF.md, docs, evidence) with the old->new commit map in the issue.

5. SLIMNESS POLICY
- Evidence, raw captures, logs, benchmark PDFs and large data live OUTSIDE this repository: in huashengshao/iregi-rust-evidence (create it if absent) or as GitHub release assets. Here: source, tests, small fixtures, Typst sources, short summaries linking to the evidence. No Symbolica copy in the archive.
- .gitignore lists every path where captures, raw JSON/JSONL, logs and generated PDFs are written, including docs/benchmark.pdf and docs/data/results.json. We commit the initial entries; when your code writes such output to a new path, add it to .gitignore in the same PR.
- Blob-size guard. Policy: no push adds a blob above MAX_BLOB_BYTES = 1 MiB unless its exact path is in .blob-size-allowlist (one exact path per line; justify additions in the PR), and no push adds a blob under dependencies/symbolica/, dependencies/symbolica-patches/ or dependencies/oneloopmaster/. This stated policy value goes into AGENTS.md; change it only by PR. We commit the hook as .githooks/pre-push (executable) and .blob-size-allowlist to main before the freeze is announced, the allowlist seeded with these lines (we re-check them at the head when we commit it):
    crates/iregi-masters/src/generated/d0.rs
    crates/iregi-masters/src/generated/c0.rs
    python/tirust/data/render/fonts/NewCMMath-Bold.otf
    python/tirust/data/render/fonts/NewCMMath-Book.otf
    python/tirust/data/render/fonts/NewCMMath-Regular.otf
  Once they are on main, run git config core.hooksPath .githooks in every clone. The hook:
    #!/bin/sh
    MAX_BLOB_BYTES=1048576   # policy value, see AGENTS.md
    remote=$1
    t='FREEZE: history rewrite'
    open=$(gh issue list --repo huashengshao/iregi-rust --state open --limit 1000 \
      --json title --jq ".[] | select(.title == \"$t\") | .title" </dev/null) ||
      { echo 'pre-push: freeze check failed (gh installed and authenticated? issues enabled?)' >&2; exit 1; }
    [ -z "$open" ] || { echo 'pre-push: FREEZE open, keep work local' >&2; exit 1; }
    z=0000000000000000000000000000000000000000
    while read -r lref lsha rref rsha; do
      [ "$lsha" = "$z" ] && continue
      if [ "$rsha" = "$z" ]; then set -- "$lsha" --not --remotes="$remote"; else set -- "$rsha..$lsha"; fi
      git rev-list --objects "$@" >/dev/null || { echo 'pre-push: rev-list failed' >&2; exit 1; }
      git rev-list --objects "$@" |
        git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' |
        awk -v max="$MAX_BLOB_BYTES" '
          FILENAME == ARGV[1] { ok[$0] = 1; next }
          $1 == "blob" { p = $0; sub(/^[^ ]+ [^ ]+ /, "", p)
            if (p ~ /^dependencies\/(symbolica|symbolica-patches|oneloopmaster)\//) {
              print "pre-push: " p " is under a path stripped by the rewrite"; bad = 1 }
            else if ($2 > max && !(p in ok)) { print "pre-push: " p " is " $2 " bytes > " max; bad = 1 } }
          END { exit bad }' .blob-size-allowlist - >&2 || exit 1
    done
    exit 0
- Update AGENTS.md by PR (your first PR): pushing to origin/main is replaced by section 0's rule; committing docs/benchmark.pdf each milestone becomes: build it locally (git-ignored), publish it as a release asset (gh release create tirust-bench-<date> docs/benchmark.pdf) or in the archive, link that; raw evidence is kept in the archive; add sections 0-6 of this prompt verbatim (or commit them as docs/PYAMPLICOL-COLLABORATION.md and reference it from the first line of AGENTS.md as binding).

6. TALKING TO US
- Questions, proposals and conflicts go to an issue here that mentions @ValentinHirschi, labelled pyamplicol plus question, api-change, pins or freeze as relevant (create missing labels with gh label create).
- Open one before touching what we maintain (Cargo manifests, Cargo.lock, pins, LICENSE, .githooks, section 3 APIs), when main fails your tests, when you need a Symbolica or oneloopmaster change (we take it upstream), or when a plan conflicts with section 2.
```
