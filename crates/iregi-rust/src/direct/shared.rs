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
//!
//! Integrals of an amplitude need a subset at different coefficient ranks (a
//! pinched box at rank 4, the boxes of its parents at rank 5). The store
//! therefore also carries a rank floor per subset size, the amplitude's
//! highest: every injected box or triangle is computed at the floor and each
//! integral injects the leading part its layout needs (layouts are
//! rank-major). The floor applies with sharing on and off, so both stay
//! bit-identical.
//!
//! The Cartesian moments gathered from a subset's coefficients can repeat
//! too: a pentagon integral gathers its Gram-free pentagon, and the hexagon
//! containing it gathers the same pentagon again. Moments are kept under a
//! key holding everything the gather reads: the Laurent slots it computes
//! (all four, or two on IR-free lanes), the gather scale, the rank, the
//! momenta and origin and the coefficient values of those slots, bit for
//! bit. A stored entry is therefore exactly what the gather would compute,
//! wherever its coefficients came from. Building and storing such keys
//! costs about as much as a small gather, so only subsets of a size and rank
//! that some integral gathers inside a larger one are shared.
//!
//! Double-double executors (partial rescues, the double-double tier and its
//! rotation check) share only their scalar masters, in a map of their own:
//! oneloopmaster's native box, triangle, bubble and tadpole values keyed by
//! the family, the arity and both limbs of every argument, bit for bit. The
//! arguments are taken in the executor's own propagator order, so a stored
//! master is exactly what that executor's kernel call returns.
use symbolica::domains::float::{Complex, DoubleFloat};
use std::{
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

/// Binary64 coefficient values (`[re, im]` of the four Laurent slots) of one
/// subset, or the refusal message of its computation.
pub(crate) type SharedValue = Result<Vec<[f64; 8]>, String>;

/// Binary64 scalar masters of one family: `re, im` bits of each argument
/// (`W / 2` of them, mu^2 last) to `re, im` of `[finite, single, double]`.
/// Keys of exactly the family's size keep the table small (one access per
/// lookup, no allocation per entry).
type MasterMap<const W: usize> = Mutex<HashMap<[u64; W], [f64; 6], BuildHasherDefault<KeyHasher>>>;

/// Most argument words of a scalar master key (a box: ten arguments and mu^2).
pub(crate) const MASTER_KEY_WORDS: usize = 22;

fn master_in<const W: usize>(map: &MasterMap<W>, words: &[u64]) -> Option<[f64; 6]> {
    let key: &[u64; W] = words.try_into().ok()?;
    map.lock().unwrap_or_else(|e| e.into_inner()).get(key).copied()
}

fn put_master_in<const W: usize>(map: &MasterMap<W>, words: &[u64], value: [f64; 6]) {
    if let Ok(key) = <[u64; W]>::try_from(words) {
        map.lock().unwrap_or_else(|e| e.into_inner()).insert(key, value);
    }
}

/// A fast hasher for the store's long exact keys: a match is decided by key
/// equality, the hash only spreads the keys over the table.
#[derive(Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        let mut h = self.0;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^ (h >> 33)
    }

    fn write(&mut self, bytes: &[u8]) {
        // Four independent lanes over 32-byte blocks (a key is hashed in one
        // call): one multiply chain per word would make hashing the long
        // moment keys latency-bound.
        const K: u64 = 0x517c_c1b7_2722_0a95;
        let mut blocks = bytes.chunks_exact(32);
        let mut lanes = [self.0, self.0 ^ 0x9e37_79b9_7f4a_7c15, self.0 ^ 0x6a09_e667_f3bc_c908, self.0 ^ 0xbb67_ae85_84ca_a73b];
        let mut any = false;
        for block in &mut blocks {
            any = true;
            for (lane, word) in lanes.iter_mut().zip(block.chunks_exact(8)) {
                let word = u64::from_le_bytes(word.try_into().unwrap());
                *lane = (lane.rotate_left(5) ^ word).wrapping_mul(K);
            }
        }
        if any {
            self.0 = lanes[0] ^ lanes[1].rotate_left(17) ^ lanes[2].rotate_left(31) ^ lanes[3].rotate_left(47);
        }
        let mut words = blocks.remainder().chunks_exact(8);
        for word in &mut words {
            self.write_u64(u64::from_le_bytes(word.try_into().unwrap()));
        }
        for &byte in words.remainder() {
            self.write_u64(u64::from(byte));
        }
    }

    fn write_u64(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }

    fn write_usize(&mut self, word: usize) {
        self.write_u64(word as u64);
    }
}

/// Results shared by every workspace holding the same store.
#[derive(Debug, Default)]
pub(crate) struct SubsetStore {
    map: Mutex<HashMap<Vec<u64>, SharedValue, BuildHasherDefault<KeyHasher>>>,
    /// Binary64 scalar masters per family (tadpole, bubble, triangle, box),
    /// counted with the coefficient results.
    tadpoles: MasterMap<4>,
    bubbles: MasterMap<8>,
    triangles: MasterMap<14>,
    boxes: MasterMap<22>,
    /// Gathered moments: the computed Laurent slots (`re, im`) of each moment.
    moments: Mutex<HashMap<Vec<u64>, Vec<f64>, BuildHasherDefault<KeyHasher>>>,
    /// Double-double scalar masters (oneloopmaster's native kernels,
    /// `[finite, single, double]`) by family and arguments, limb for limb:
    /// the partial rescues, rotation checks and double-double tiers of an
    /// amplitude evaluate the same box and triangle masters many times.
    dd_masters: Mutex<HashMap<Vec<u64>, [Complex<DoubleFloat>; 3], BuildHasherDefault<KeyHasher>>>,
    /// Sharing on (`false`: the floors still apply, nothing is shared).
    enabled: std::sync::atomic::AtomicBool,
    /// Coefficient rank floor per subset size (number of propagators).
    floors: Mutex<Vec<u16>>,
    /// `(size, rank)` of the gathers whose moments are shared.
    moment_shapes: Mutex<Vec<(usize, u16)>>,
    hits: AtomicU64,
    misses: AtomicU64,
    moment_hits: AtomicU64,
    moment_misses: AtomicU64,
    dd_master_hits: AtomicU64,
    dd_master_misses: AtomicU64,
}

impl SubsetStore {
    pub(crate) fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Raise the floor of `size`-propagator subsets to `rank`.
    pub(crate) fn raise_floor(&self, size: usize, rank: u16) {
        let mut floors = self.floors.lock().unwrap_or_else(|e| e.into_inner());
        if floors.len() <= size {
            floors.resize(size + 1, 0);
        }
        floors[size] = floors[size].max(rank);
    }

    /// The coefficient rank at which a `size`-propagator subset needed at
    /// `rank` is computed.
    pub(crate) fn rank_for(&self, size: usize, rank: u16) -> u16 {
        let floors = self.floors.lock().unwrap_or_else(|e| e.into_inner());
        rank.max(floors.get(size).copied().unwrap_or(0))
    }

    /// Share the moments of `size`-propagator subsets gathered at `rank`.
    pub(crate) fn share_moments(&self, size: usize, rank: u16) {
        let mut shapes = self.moment_shapes.lock().unwrap_or_else(|e| e.into_inner());
        if !shapes.contains(&(size, rank)) {
            shapes.push((size, rank));
        }
    }

    /// Whether the moments of `size`-propagator subsets gathered at `rank`
    /// are shared.
    pub(crate) fn shares_moments(&self, size: usize, rank: u16) -> bool {
        self.moment_shapes.lock().unwrap_or_else(|e| e.into_inner()).contains(&(size, rank))
    }

    /// The stored result for `key` (a hit), if any.
    pub(crate) fn get(&self, key: &[u64]) -> Option<SharedValue> {
        let value = self.map.lock().unwrap_or_else(|e| e.into_inner()).get(key).cloned();
        if value.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
        value
    }

    /// Pass the stored result for `key` to `read` (a hit), without copying it.
    pub(crate) fn get_with<R>(&self, key: &[u64], read: impl FnOnce(&SharedValue) -> R) -> Option<R> {
        let map = self.map.lock().unwrap_or_else(|e| e.into_inner());
        let value = map.get(key)?;
        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(read(value))
    }

    /// The stored binary64 scalar master of `family` for the argument bits
    /// `words` (a hit), if any.
    pub(crate) fn master(&self, family: usize, words: &[u64]) -> Option<[f64; 6]> {
        let value = match family {
            0 => master_in(&self.tadpoles, words),
            1 => master_in(&self.bubbles, words),
            2 => master_in(&self.triangles, words),
            _ => master_in(&self.boxes, words),
        };
        if value.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
        value
    }

    /// Store a freshly evaluated binary64 scalar master (a miss).
    pub(crate) fn put_master(&self, family: usize, words: &[u64], value: [f64; 6]) {
        self.misses.fetch_add(1, Ordering::Relaxed);
        match family {
            0 => put_master_in(&self.tadpoles, words, value),
            1 => put_master_in(&self.bubbles, words, value),
            2 => put_master_in(&self.triangles, words, value),
            _ => put_master_in(&self.boxes, words, value),
        }
    }

    /// Store a freshly computed result (a miss).
    pub(crate) fn put(&self, key: Vec<u64>, value: SharedValue) {
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.map.lock().unwrap_or_else(|e| e.into_inner()).insert(key, value);
    }

    /// Pass the stored Cartesian moments for `key` to `write`; a hit when
    /// they exist and `write` accepts them.
    pub(crate) fn moments_with(&self, key: &[u64], write: impl FnOnce(&[f64]) -> bool) -> bool {
        let hit = match self.moments.lock().unwrap_or_else(|e| e.into_inner()).get(key) {
            Some(moments) => write(moments),
            None => false,
        };
        if hit {
            self.moment_hits.fetch_add(1, Ordering::Relaxed);
        }
        hit
    }

    /// Store freshly gathered Cartesian moments (a miss).
    pub(crate) fn put_moments(&self, key: Vec<u64>, moments: Vec<f64>) {
        self.moment_misses.fetch_add(1, Ordering::Relaxed);
        self.moments.lock().unwrap_or_else(|e| e.into_inner()).insert(key, moments);
    }

    /// Pass the stored double-double master for `key` to `write`; a hit when
    /// it exists and `write` accepts it.
    pub(crate) fn dd_master_with(&self, key: &[u64], write: impl FnOnce(&[Complex<DoubleFloat>; 3]) -> bool) -> bool {
        let hit = match self.dd_masters.lock().unwrap_or_else(|e| e.into_inner()).get(key) {
            Some(value) => write(value),
            None => false,
        };
        if hit {
            self.dd_master_hits.fetch_add(1, Ordering::Relaxed);
        }
        hit
    }

    /// Store a freshly evaluated double-double master (a miss).
    pub(crate) fn put_dd_master(&self, key: Vec<u64>, value: [Complex<DoubleFloat>; 3]) {
        self.dd_master_misses.fetch_add(1, Ordering::Relaxed);
        self.dd_masters.lock().unwrap_or_else(|e| e.into_inner()).insert(key, value);
    }

    /// Forget every result and reset the counters.
    pub(crate) fn clear(&self) {
        self.map.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.tadpoles.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.bubbles.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.triangles.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.boxes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.moments.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.dd_masters.lock().unwrap_or_else(|e| e.into_inner()).clear();
        for counter in [
            &self.hits,
            &self.misses,
            &self.moment_hits,
            &self.moment_misses,
            &self.dd_master_hits,
            &self.dd_master_misses,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
    }

    /// `(hits, misses)` of coefficient results since the last clear.
    pub(crate) fn counts(&self) -> (u64, u64) {
        (self.hits.load(Ordering::Relaxed), self.misses.load(Ordering::Relaxed))
    }

    /// `(hits, misses)` of gathered moments since the last clear.
    pub(crate) fn moment_counts(&self) -> (u64, u64) {
        (self.moment_hits.load(Ordering::Relaxed), self.moment_misses.load(Ordering::Relaxed))
    }

    /// `(hits, misses)` of double-double scalar masters since the last clear.
    pub(crate) fn dd_master_counts(&self) -> (u64, u64) {
        (self.dd_master_hits.load(Ordering::Relaxed), self.dd_master_misses.load(Ordering::Relaxed))
    }
}
