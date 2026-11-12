# Scoped Symbolica library activation

TIRust's Rust core (`iregi_rust`, Cargo package `iregi-rust`) contains a signed
Symbolica library key issued for that crate. Its signed offline expiry is
1 March 2028. The registration and unlock handles are private in
`crates/iregi-rust/src/license.rs`; this is library activation, not a global
application or personal license.

Core plan preparation and evaluation hold a Symbolica `LibraryUnlockGuard`
for the operation. Calls into `iregi-masters` and oneloopmaster on that call
chain inherit the same scope. Symbolica propagates the guard to its own
workers, and each external thread enters through the guarded TIRust API.
Guards also cover the shared direct executor used by embedding adapters.
They are dropped on normal return, errors and unwinding.

This does not globally unlock unrelated host Symbolica operations. Separate
oneloopmaster calls and graph/numerator preparation performed outside the
TIRust core retain their own licensing requirements. The library-key agreement
with Symbolica determines the authorized scope. Expiry/revocation and renewal
follow Symbolica's library-key behavior; no license validation is bypassed.

The focused integration test launches fresh processes with `SYMBOLICA_LICENSE`
removed and an isolated `SYMBOLICA_LOCK_DIR`. It checks the actual direct,
generic, embedded and MPFR paths, master-provider inheritance followed by a
real oneloopmaster evaluation, independent caller threads and panic unwinding.
The host's two-thread request remains restricted before and after each core
operation; the downstream callback sees the scoped authorization.

```sh
cargo test -p iregi-rust --test oem_library -- --test-threads=1
```

No dependency revision, numerical algorithm, precision, stability threshold,
convention or public API signature changes as part of activation. An embedding
application must rebuild against the signed TIRust revision; an old installed
shared library does not acquire the new registration automatically.
