//! Per-subset results shared between the integrals of an amplitude.
//!
//! The expensive per-lane work of a binary64 direct evaluation on difficult
//! kinematics is per box or triangle: the partial rescue computes a refused
//! one in double-double, the method-choice retries expand it by the
//! small-Gram expansions. The integrals of an amplitude contain the same
//! boxes and triangles (a pinched integral's are its parent's), so the same
//! computations repeat. A [`SubsetStore`] keeps each such result under an
//! exact key: what was computed (partial rescue or expansion, with the
//! method's settings), the subset's size, coefficient rank and value count,
//! and the subset's propagators bit for bit (shifts, masses) with mu. A
//! result is a pure function of its key, so every integral receives exactly
//! the values it would have computed itself: results are bit-identical with
//! and without sharing.
use std::{
    collections::HashMap,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

/// Binary64 coefficient values (`[re, im]` of the four Laurent slots) of one
/// subset, or the refusal message of its computation.
pub(crate) type SharedValue = Result<Vec<[f64; 8]>, String>;

/// Results shared by every workspace holding the same store.
#[derive(Debug, Default)]
pub(crate) struct SubsetStore {
    map: Mutex<HashMap<Vec<u64>, SharedValue>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl SubsetStore {
    /// The stored result for `key` (a hit), if any.
    pub(crate) fn get(&self, key: &[u64]) -> Option<SharedValue> {
        let value = self.map.lock().unwrap_or_else(|e| e.into_inner()).get(key).cloned();
        if value.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
        value
    }

    /// Store a freshly computed result (a miss).
    pub(crate) fn put(&self, key: Vec<u64>, value: SharedValue) {
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.map.lock().unwrap_or_else(|e| e.into_inner()).insert(key, value);
    }

    /// Forget every result and reset the counters.
    pub(crate) fn clear(&self) {
        self.map.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// `(hits, misses)` since the last clear.
    pub(crate) fn counts(&self) -> (u64, u64) {
        (self.hits.load(Ordering::Relaxed), self.misses.load(Ordering::Relaxed))
    }
}
