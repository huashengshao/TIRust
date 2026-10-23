# Exact amplitude-variable collector qualification

The private candidate replaces repeated discovery of amplitude variables in
IF-free expressions with Symbolica's `get_all_indeterminates(true)`. It keeps
only variable atoms and inserts them into the original ordered set. Whole
`IF` expressions remain separate generators, handled before the fast path.
Numeric coefficients remain opaque. Conditional recursion and
`depends_on_amplitudes` are unchanged.

This distinction matters for rational-polynomial numbers: Symbolica's
`get_all_symbols(false)` inspects their variables, whereas the original
amplitude-variable collector ignores them. The candidate therefore uses
indeterminate discovery, whose `Num` arm is empty. Function names are also
filtered out; function arguments and both power operands remain visible.
The relevant existing APIs are in Symbolica `src/atom/core.rs` and
`src/id.rs`. This is the same API distinction checked for the earlier alias-use
collector, documented in `highpoint-alias-use-dedup.md`; no algebra operation
is implemented independently.

The exact probe passes 48 complete expected-set comparisons: 24 cases with
empty and preseeded sets. They cover repeated occurrences, function-name
collisions, power exponents, opaque polynomial coefficients, whole and nested
IFs, condition-only dependence, hidden coefficient dependence inside an IF
condition, inactive singular branches, and composite amplitude-map keys. The
probe asserts that IF and repeated-occurrence fixtures survive Symbolica
normalization. Arbitrary preseeded entries remain present.

Another 34 cases compare complete canonical amplitude expressions or complete
error strings, including the previously qualified 20 amplitude witnesses.
The legacy amplitude path recursively uses its own unchanged collector.
Both verification modes are mandatory in the standalone probe, preventing
an accidental legacy-versus-legacy comparison. The complete 38 stability and
35 execution tests also pass with both verification modes enabled. This
establishes the recorded semantic coverage; a full high-point numerical or
performance qualification is separate.

The independent artifact audit verifies 221 hash bindings, all witness
identities and classifications, and test membership against the full
174-test listing. The frozen historical build uses 112 Rust dependency
objects, three native archives, and 20 guarded toolchain dependencies. It is
the matched historical development-profile build, not the optimized current
workspace. Final links strip debug sections to reduce disk use; codegen,
assertion, and overflow settings remain recorded explicitly.

Two failed intermediate build records remain visible. A storage-priority
coordination pause exhausted the original 240-second build budget after
core/public compilation; no later children were started. A separate guarded
core rebuild then failed an unnecessarily strict whole-archive byte-equality
check because the output filename changed. Independent inspection found all
16 compiled-object payloads and `lib.rmeta` identical, with the differing
`lib.rmeta-link` containing the changed filenames. Final executables use the
freshly rebuilt, source-bound archive directly. No numerical tolerance was
changed, and no whole-archive reproducibility claim is made.

Evidence under `validation/public-highpoint/`:

- `amplitude-vars-build-final.json`, SHA256
  `7191ca19767f997ee507d01d9dbbd371467ae396a48829fd31f34f36c869e764`.
- `amplitude-vars-checks.json`, SHA256
  `4d2fb64bc0039b114ce122f9044393613bcb7d902b9e9fa70aae5a0b97553a4e`.
- `amplitude-vars-independent-audit.json`, SHA256
  `9053d3348255952b7adf597ca3b183c305bcd7bfaa316574267bbc0f0811c4d8`.
- `amplitude-vars-adoption.patch`, SHA256
  `03ab8645391035b3dd8735d48ec134f33278b7b38fb4193ebb175066904dae23`.

The adoption patch is a separately prepared, 24-line change without diagnostic
modes, duplicate legacy code, phase counters, or threshold changes. Its
manifest records the production source it was prepared against; preparation
does not apply it. Public N6 capture artifacts and their eventual outcome
must be assessed independently of these exact witnesses.

## Bounded public N6 outcome

The candidate's unchanged N6 rank-seven fixture still times out during
workspace preparation. The process was killed after 601.3289 seconds against
the fixed 600-second limit; it did not enter numerical reduction and produced
no successful arrays. All frozen source, input, executable, dependency and
prerequisite evidence guards passed. Peak child RSS was 11,714,960 KiB.

| Diagnostic phase | Previous collector+nway candidate | Amplitude-variable candidate |
| --- | ---: | ---: |
| Structural plan | 41.7608 s | 56.5287 s |
| Reverse alias uses | 10.5369 s | 12.9556 s |
| Local stability checks | 185.9402 s | 204.1260 s |
| Primal amplitudes | 346.5734 s | 293.1038 s |
| Master activations | Entered, timed out | Entered, timed out |

Both runs retain 31,670 aliases, 25,700 reachable aliases, 19,254 stability
checks and 576,834 local monitor roots. The candidate's primal dependency scan
count is unchanged at 951,437. The lower observed primal phase time is a
diagnostic finding, not a controlled timing qualification: other phases became
slower, final links differ in debug-section retention, and no paired timing
protocol was run. The new outcome does not establish N6/N7 numerical coverage
or justify increasing the resource limit. Structural rank lowering is the
next research direction in `collier-1.2.10-audit.md`.

The complete guarded capture is
`n6-rank7-amplitude-vars-128-53-qualification.json`, SHA256
`9262c510227e6ff94154715fc26f207412c6dddd79c49d854ebd77395868be83`.
Its run manifest has SHA256
`dd545e1e8cb092e2492f1c394c60f2e78fccc3670ee0c1a51015ab4e3c78f320`;
the retained log has SHA256
`65d8a407946296c4396b94837913603d1d10d48f2fd3442119c96d65234c007f`.
The earlier 180/300/600-second failures remain preserved.

After restoring the exact frozen prerequisites and sourcing `scripts/env.sh`
plus the private license environment, reproduction uses a fresh capture name:

```sh
python3 -B validation/public-highpoint/run_amplitude_variables_full.py \
  NEW_CAPTURE_NAME --cpu 14 --limit 600
python3 -B validation/public-highpoint/qualify_amplitude_variables_full.py \
  NEW_CAPTURE_NAME
```

The runner refuses to overwrite evidence, requires the successful exact and
semantic qualification, validates all complete-array claims, and preserves
timeout or malformed-ledger failures explicitly.
