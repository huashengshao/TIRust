//! Scoped Symbolica authorization for TIRust operations.
//!
//! The signed library key belongs to the Rust crate `iregi_rust`. It is not an
//! application license: callers' unrelated Symbolica work keeps its own licensing state.
use std::sync::LazyLock;

use symbolica::license::{LibraryUnlock, LibraryUnlockGuard};

// Issued for TIRust by Symbolica; the signed offline expiry is 2028-03-01.
// Keep both the registration and its guard private to the reduction library.
static LIBRARY: LazyLock<LibraryUnlock> = LazyLock::new(|| {
    symbolica::set_library_key!(
        "SU-466-iregi_rust-2028.03.01-YBAYFKTVHOSZOKVGXP43BRKRMWA6VBAZNTILWPM2G3LG7L7INRWDV3I6TZOOBAL6OR44JYEHNNORCYAXPTERCNXULDZDVMTGTKGUYAY"
    )
    .expect("TIRust's embedded Symbolica library key must be valid for iregi_rust")
});

/// Hold this guard for the whole operation, including downstream masters.
/// Symbolica propagates its scoped authorization to its own worker threads;
/// external threads enter separately through TIRust's guarded entry points.
#[inline]
pub(crate) fn unlock() -> LibraryUnlockGuard {
    LIBRARY.unlock()
}
