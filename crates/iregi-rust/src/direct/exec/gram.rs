//! Small-Gram expansion of box coefficients (Denner and Dittmaier,
//! hep-ph/0509141, sect. "Reduction for small Gram determinant", eqs.
//! (recurGram1) and (recurGram2); COLLIER's `CalcDg`).
//!
//! For a box with momenta `p_1..p_3` (relative to its first member), Gram
//! matrix `Z = (2 p_i p_j)`, `Delta = det Z` and `f_i = p_i^2 - m_i^2 + m_0^2`,
//! the PV relations contracted with the adjugate `Zadj` give, with
//! `(Zadj f)_j = sum_n Zadj_jn f_n` maximal and `Zadj_kl` maximal,
//!
//!   C(0, k) = [ sum_n Zadj_jn (Shat_n(0, k) - 2 k_n C(1, k - e_n))
//!               - Delta C(0, k + e_j) ] / (Zadj f)_j,
//!
//!   4 ((l + |k|) - eps) Zadj_kl C(l, k) = - Delta C(l-1, k + e_k + e_l)
//!       + 2 Zadj_kl C0(l-1, k) + Xtilde_kl C(l-1, k)
//!       + sum_n (Zadj_nl Shat_n(l-1, k + e_k) - Zadj_kl Shat_n(l-1, k + e_n))
//!       - sum_nm Zadj2_(kn)(lm) [ f_n Shat_m(l-1, k) + 2 k_n Shat_m(l, k - e_n)
//!           - 2 f_n k_m C(l, k - e_m) - 2 f_m k_n C(l, k - e_n)
//!           - 4 (k_n k_m - delta_nm k_n) C(l+1, k - e_n - e_m) ],
//!
//! for `C(lambda, kappa)` in the engine's normalization (`lambda` metric
//! pairs, momentum multiplicities `kappa`). `Shat_n` is the triangle without
//! member `n` minus the shifted triangle without the first member (`C0`),
//! `Zadj2` the second adjugate and `Xtilde_kl = 2 m_0^2 Zadj_kl +
//! sum_nm Zadj2_(kn)(lm) f_n f_m`. Dropping the `Delta` terms gives every
//! coefficient from triangles of the same rank; each further pass puts the
//! previous pass into the `Delta` terms and gains a factor `Delta /
//! (Zadj f)_j`. With triangles of rank `r + G` the rank-`r` coefficients
//! carry `G` corrections. The scalar `D0` is the master itself. When it
//! does not converge, the expansion for small Gram and small modified Cayley
//! determinants (sect. 3.5) is tried, then the alternative PV reduction
//! through the modified Cayley matrix (sect. 3.2), which is exact.

use std::collections::HashMap;

use iregi_masters::NativeFloat;
use symbolica::{domains::float::Complex, evaluate::EvaluationDomain};

use super::{DirectExecutor, DirectMasters};
use crate::direct::{
    numeric::{C, L4, cadd, cinv, cmul, cnorm2, cscale, csub, czero, div_dimension, lfma, lzero, mul_dimension},
    plan::{DirectPlanData, SubsetKind},
};
use crate::{KinematicsBatch, LaneStatus, ReductionSettings};

/// Expansion executors of a binary64 executor, by subset size, coefficient
/// rank and boost.
#[derive(Clone)]
pub(crate) struct GramRescue<T: NativeFloat> {
    executors: HashMap<(usize, u16, u16), DirectExecutor<T>>,
    /// Boosts tried in order: a smaller one first (fewer high-rank triangles
    /// for its gate to refuse), the largest one only where it fails.
    boosts: Vec<u16>,
    /// Alternative PV reduction only (no expansions).
    alternative: bool,
}

impl<T: NativeFloat> GramRescue<T> {
    pub(crate) fn new(boost: u16) -> Self {
        let mut boosts = vec![boost.min(FIRST_BOOST), boost];
        boosts.dedup();
        Self { executors: HashMap::new(), boosts, alternative: false }
    }

    /// The alternative PV reduction alone: triangles of the box's own rank
    /// suffice (boost 1 keeps the expansion bookkeeping valid).
    pub(crate) fn alternative() -> Self {
        Self { executors: HashMap::new(), boosts: vec![1], alternative: true }
    }
}

/// First boost tried (see `GramRescue::boosts`).
const FIRST_BOOST: u16 = 8;

type Key = (u16, [u16; 3]);

/// Largest a priori error estimate of the alternative PV stage (the binary64
/// retries accept rotated copies within 2^-36).
const ALTERNATIVE_PV_LIMIT: f64 = 1.0 / (1u64 << 36) as f64;

/// Largest accepted relative change of the last expansion pass.
const EXPANSION_TOLERANCE: f64 = 1.0 / (1u64 << 30) as f64;

fn rank_of(key: &Key) -> u16 {
    2 * key.0 + key.1.iter().sum::<u16>()
}

/// Every `kappa` with `|kappa| = total` over `nm` (2 or 3) momenta, padded
/// with zeros to three components.
fn compositions(total: u16, nm: usize) -> Vec<[u16; 3]> {
    let mut out = Vec::new();
    if nm == 2 {
        for a in 0..=total {
            out.push([a, total - a, 0]);
        }
        return out;
    }
    for a in 0..=total {
        for b in 0..=total - a {
            out.push([a, b, total - a - b]);
        }
    }
    out
}

fn binomial(n: u16, k: u16) -> u64 {
    (0..u64::from(k)).fold(1u64, |acc, i| acc * (u64::from(n) - i) / (i + 1))
}

/// `-sum_{m,n} eps_{ikm} eps_{jln} Z_mn`: the second adjugate of a 3x3
/// matrix in the paper's convention (its relations (Zadjadjrelation0-2)
/// fix the sign; see the test below).
fn second_adjugate<T: NativeFloat>(z: &[[C<T>; 3]; 3], nm: usize, i: usize, k: usize, j: usize, l: usize) -> Option<C<T>> {
    if i == k || j == l {
        return None;
    }
    if nm == 2 {
        // The paper's N = 2 case: d_il d_kj - d_ij d_kl (i != k, j != l).
        let one = C::new(z[0][0].re.one(), z[0][0].re.zero());
        return Some(if i == l && k == j { one } else { csub(&czero(&one.re), &one) });
    }
    let sign = |a: usize, b: usize| -> bool {
        // eps_{a b c} with c the third index: +1 for cyclic (0,1),(1,2),(2,0).
        (b + 3 - a) % 3 == 1
    };
    let m = 3 - i - k;
    let n = 3 - j - l;
    let value = z[m][n].clone();
    Some(if sign(i, k) == sign(j, l) { csub(&czero(&value.re), &value) } else { value })
}

impl<T: NativeFloat> DirectExecutor<T>
where
    Complex<T>: EvaluationDomain,
{
    /// Evaluate every expansion box of this lane by its auxiliary executor
    /// and inject its coefficients.
    pub(super) fn inject_gram(&mut self, mu: &C<T>) -> Result<(), String> {
        let plan = self.plan.clone();
        let boxes = self.gram_external.clone();
        let mut gram = self.gram.take().ok_or("no small-Gram expansion")?;
        let outcome = (|| -> Result<(), String> {
            for &id in &boxes {
                let subset = &plan.subsets[id];
                let rank = subset.coefficient_rank.max(0) as u16;
                let size = subset.members.len();
                let mut failure = String::new();
                let mut done = false;
                for boost in gram.boosts.clone() {
                let executor = match gram.executors.entry((size, rank, boost)) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let aux = DirectPlanData::with_top_child_boost(ReductionSettings::new(size, rank), boost)
                            .map_err(|e| e.to_string())?;
                        let mut executor = DirectExecutor::<T>::new(
                            std::sync::Arc::new(aux),
                            self.prototype.clone(),
                            DirectMasters::Native,
                        )
                        .map_err(|e| e.to_string())?;
                        // Its triangles carry `boost` extra ranks whose errors
                        // enter only Delta-suppressed terms (COLLIER relaxes them
                        // likewise); the caller's a posteriori check decides.
                        executor.set_stability_threshold(self.stability_threshold.clone() * &self.stability_threshold);
                        executor.gram_top = Some(rank);
                        executor.gram_alt_only = gram.alternative;
                        entry.insert(executor)
                    }
                };
                let mut values = Vec::with_capacity(21);
                for &member in &subset.members {
                    values.extend(self.shifts[member].iter().cloned());
                }
                for &member in &subset.members {
                    values.push(self.masses[member].clone());
                }
                values.push(mu.clone());
                let result = executor
                    .reduce_batch(&KinematicsBatch { rows: 1, components: 1, values })
                    .map_err(|e| e.to_string())?;
                if result.statuses[0] != LaneStatus::Success {
                    failure = format!("small-Gram expansion of {:?}: {:?}", subset.members, result.statuses[0]);
                    continue;
                }
                let coefficients = executor.top_coefficients().ok_or("small-Gram expansion: no coefficients")?;
                let count = plan.coefficient_counts[id] as usize;
                if coefficients.len() < count {
                    return Err("small-Gram expansion: coefficient layouts differ".into());
                }
                let start = subset.master.ok_or("small-Gram expansion: no master")? as usize;
                self.values[start..start + count].clone_from_slice(&coefficients[..count]);
                done = true;
                break;
                }
                if !done {
                    return Err(failure);
                }
            }
            Ok(())
        })();
        self.gram = Some(gram);
        outcome
    }

    /// Fill the top box's coefficients up to rank `target` from its
    /// triangles by the small-Gram expansion (auxiliary executors).
    pub(super) fn expand_top(&mut self, target: u16) -> Result<(), String> {
        let plan = self.plan.clone();
        let id = plan.subsets.len() - 1;
        let subset = &plan.subsets[id];
        if subset.kind != SubsetKind::Pv || !(3..=4).contains(&subset.members.len()) {
            return Err("small-Gram expansion: the top subset is not a PV triangle or box".into());
        }
        let nm = subset.members.len() - 1;
        let n = plan.settings.propagators;
        let members = &subset.members;
        let origin = members[0];
        let zero = czero(&self.prototype);
        let dist = |i: usize, j: usize| self.distances[i * n + j].clone();
        let z: [[C<T>; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if i >= nm || j >= nm {
                    return zero.clone();
                }
                let (a, b) = (members[i + 1], members[j + 1]);
                csub(&cadd(&dist(origin, a), &dist(origin, b)), &dist(a, b))
            })
        });
        let geometry = subset.pv.unwrap();
        let f: [C<T>; 3] = std::array::from_fn(|i| {
            if i < nm { self.geometry[geometry.f as usize + i].clone() } else { zero.clone() }
        });
        let m02 = self.masses[origin].clone();
        // Adjugate (Z symmetric) and determinant.
        let adj: [[C<T>; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if nm == 2 {
                    return match (i, j) {
                        (0, 0) => z[1][1].clone(),
                        (1, 1) => z[0][0].clone(),
                        (0, 1) | (1, 0) => csub(&zero, &z[j][i]),
                        _ => zero.clone(),
                    };
                }
                let (a, b, c, d) = ((j + 1) % 3, (j + 2) % 3, (i + 1) % 3, (i + 2) % 3);
                csub(&cmul(&z[a][c], &z[b][d]), &cmul(&z[a][d], &z[b][c]))
            })
        });
        let mut delta = zero.clone();
        for j in 0..nm {
            delta = cadd(&delta, &cmul(&z[0][j], &adj[j][0]));
        }
        let adjf: [C<T>; 3] = std::array::from_fn(|j| {
            (0..nm).fold(zero.clone(), |acc, m| cadd(&acc, &cmul(&adj[j][m], &f[m])))
        });
        let j = (0..nm)
            .max_by(|&a, &b| cnorm2(&adjf[a]).partial_cmp(&cnorm2(&adjf[b])).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let (mut k, mut l) = (0, 0);
        for a in 0..nm {
            for b in a..nm {
                if cnorm2(&adj[a][b]) > cnorm2(&adj[k][l]) {
                    (k, l) = (a, b);
                }
            }
        }
        if cnorm2(&adjf[j]).is_zero() || cnorm2(&adj[k][l]).is_zero() {
            return Err("small-Gram expansion: vanishing adjugates".into());
        }
        let inv_adjf = cinv(&adjf[j]);
        let two = self.prototype.from_usize(2);
        let four = self.prototype.from_usize(4);
        let inv_two_adj = cinv(&cscale(&adj[k][l], &two));
        // Second adjugates Zadj2_(kn)(lm) and Xtilde_kl.
        let mut pairs: Vec<(usize, usize, C<T>)> = Vec::new();
        for a in 0..nm {
            for b in 0..nm {
                if let Some(w) = second_adjugate(&z, nm, k, a, l, b) {
                    pairs.push((a, b, w));
                }
            }
        }
        let mut xtilde = cscale(&cmul(&m02, &adj[k][l]), &two);
        for (a, b, w) in &pairs {
            xtilde = cadd(&xtilde, &cmul(w, &cmul(&f[*a], &f[*b])));
        }

        // Triangle values: the pinch of member `i + 1` and of the origin.
        let lookup = |mask: u8| plan.coefficient_lookup.get(&mask);
        let mask = subset.mask;
        let children: Vec<&HashMap<(u16, Vec<u16>), u32>> = (0..nm)
            .map(|i| lookup(mask & !(1u8 << members[i + 1])).ok_or("small-Gram expansion: missing pinch"))
            .collect::<Result<_, _>>()?;
        let zero_child = lookup(mask & !(1u8 << origin)).ok_or("small-Gram expansion: missing triangle")?;
        let child_rank = children
            .iter()
            .chain(std::iter::once(&zero_child))
            .map(|layout| layout.keys().map(|(lambda, kappa)| 2 * lambda + kappa.iter().sum::<u16>()).max().unwrap_or(0))
            .min()
            .unwrap_or(0);
        if child_rank < target {
            return Err("small-Gram expansion: triangles below the requested rank".into());
        }
        let values = &self.values;
        let value = |layout: &HashMap<(u16, Vec<u16>), u32>, lambda: u16, kappa: Vec<u16>| -> Result<&L4<T>, String> {
            layout
                .get(&(lambda, kappa))
                .map(|&index| &values[index as usize])
                .ok_or_else(|| "small-Gram expansion: triangle coefficient outside its layout".to_string())
        };
        // Shifted pinch-zero values C0(lambda, kappa) in the box basis
        // (Denner, Eq. (2.9); as the PV program's `spz` instructions).
        let mut spz: HashMap<Key, L4<T>> = HashMap::new();
        for r in 0..=child_rank {
            for lambda in 0..=r / 2 {
                for kappa in compositions(r - 2 * lambda, nm) {
                    let first = kappa[0];
                    let mut acc = lzero(&zero);
                    for total in 0..=first {
                        // Distributions of `total` over the other momenta.
                        for e0 in 0..=if nm == 3 { total } else { 0 } {
                            let e1 = total - e0;
                            let (count, key) = if nm == 3 {
                                (binomial(first, total) * binomial(total, e0), vec![kappa[1] + e0, kappa[2] + e1])
                            } else {
                                (binomial(first, total), vec![kappa[1] + total])
                            };
                            let mut c = self.prototype.from_usize(count as usize);
                            if first % 2 == 1 {
                                c = -c;
                            }
                            let source = value(zero_child, lambda, key)?;
                            lfma(&mut acc, &C::new(c, self.prototype.zero()), source);
                        }
                    }
                    spz.insert((lambda, kappa), acc);
                }
            }
        }
        // Shat_n(lambda, kappa) = triangle without member n+1 (if kappa_n = 0)
        // minus the shifted pinch zero.
        let mut shat: HashMap<(usize, u16, [u16; 3]), L4<T>> = HashMap::new();
        for (&(lambda, kappa), c0) in &spz {
            for m in 0..nm {
                let mut out: L4<T> = std::array::from_fn(|slot| csub(&zero, &c0[slot]));
                if kappa[m] == 0 {
                    let reduced: Vec<u16> = (0..nm).filter(|&i| i != m).map(|i| kappa[i]).collect();
                    let source = value(children[m], lambda, reduced)?;
                    for slot in 0..4 {
                        out[slot] = cadd(&out[slot], &source[slot]);
                    }
                }
                shat.insert((m, lambda, kappa), out);
            }
        }
        let shat_of = |m: usize, lambda: u16, kappa: [u16; 3]| -> Result<&L4<T>, String> {
            shat.get(&(m, lambda, kappa)).ok_or_else(|| "small-Gram expansion: Shat outside the triangle ranks".to_string())
        };
        let d0 = self.values[subset.master.unwrap() as usize].clone();
        let real = |x: usize| C::new(self.prototype.from_usize(x), self.prototype.zero());
        let add = |kappa: [u16; 3], i: usize| -> [u16; 3] {
            let mut out = kappa;
            out[i] += 1;
            out
        };
        let sub = |kappa: [u16; 3], i: usize| -> Option<[u16; 3]> {
            (kappa[i] > 0).then(|| {
                let mut out = kappa;
                out[i] -= 1;
                out
            })
        };

        // Passes: t = 0 drops every Delta term; pass t has metric coefficients
        // up to rank R + 1 - t and momentum coefficients up to R - t.
        let passes = child_rank - target;
        let mut previous: Option<HashMap<Key, L4<T>>> = None;
        let mut last_change = f64::INFINITY;
        let mut chosen: Option<HashMap<Key, L4<T>>> = None;
        for t in 0..=passes {
            let metric_rank = child_rank + 1 - t;
            let momentum_rank = child_rank - t;
            let mut cur: HashMap<Key, L4<T>> = HashMap::new();
            cur.insert((0, [0, 0, 0]), d0.clone());
            let get = |map: &HashMap<Key, L4<T>>, key: Key| -> Result<L4<T>, String> {
                map.get(&key).cloned().ok_or_else(|| format!("small-Gram expansion: coefficient {key:?} missing"))
            };
            for r in 1..=metric_rank {
                for lambda in (1..=r / 2).rev() {
                    for kappa in compositions(r - 2 * lambda, nm) {
                        let mut acc = lzero(&zero);
                        if let Some(prev) = &previous {
                            let key = (lambda - 1, add(add(kappa, k), l));
                            lfma(&mut acc, &csub(&zero, &delta), &get(prev, key)?);
                        }
                        lfma(&mut acc, &cscale(&adj[k][l], &two), spz.get(&(lambda - 1, kappa)).ok_or("small-Gram expansion: C0 outside the triangle ranks")?);
                        lfma(&mut acc, &xtilde, &get(&cur, (lambda - 1, kappa))?);
                        for m in 0..nm {
                            lfma(&mut acc, &adj[m][l], shat_of(m, lambda - 1, add(kappa, k))?);
                            lfma(&mut acc, &csub(&zero, &adj[k][l]), shat_of(m, lambda - 1, add(kappa, m))?);
                        }
                        for (a, b, w) in &pairs {
                            let (a, b) = (*a, *b);
                            let mut bracket = lzero(&zero);
                            lfma(&mut bracket, &f[a], shat_of(b, lambda - 1, kappa)?);
                            if let Some(less) = sub(kappa, a) {
                                lfma(&mut bracket, &real(2 * kappa[a] as usize), shat_of(b, lambda, less)?);
                                let factor = cscale(&f[b], &(two.clone() * &self.prototype.from_usize(kappa[a] as usize)));
                                lfma(&mut bracket, &csub(&zero, &factor), &get(&cur, (lambda, less))?);
                            }
                            if let Some(less) = sub(kappa, b) {
                                let factor = cscale(&f[a], &(two.clone() * &self.prototype.from_usize(kappa[b] as usize)));
                                lfma(&mut bracket, &csub(&zero, &factor), &get(&cur, (lambda, less))?);
                            }
                            let multiplicity = if a == b {
                                kappa[a] as usize * (kappa[a] as usize).saturating_sub(1)
                            } else {
                                kappa[a] as usize * kappa[b] as usize
                            };
                            if multiplicity > 0 {
                                let less = sub(sub(kappa, a).unwrap(), b).unwrap();
                                let factor = cscale(&real(multiplicity), &four);
                                lfma(&mut bracket, &csub(&zero, &factor), &get(&cur, (lambda + 1, less))?);
                            }
                            lfma(&mut acc, &csub(&zero, w), &bracket);
                        }
                        let mut out: L4<T> = std::array::from_fn(|slot| cmul(&acc[slot], &inv_two_adj));
                        // Divisor (a - 2 eps), a = 2 (lambda + |kappa|) + 4 - N.
                        let a = self.prototype.from_usize(2 * (lambda as usize + kappa.iter().map(|&x| x as usize).sum::<usize>()) + 3 - nm);
                        div_dimension(&mut out, &(self.prototype.one() / &a));
                        cur.insert((lambda, kappa), out);
                    }
                }
                if r - 1 <= momentum_rank && r >= 2 {
                    for kappa in compositions(r - 1, nm) {
                        let mut acc = lzero(&zero);
                        for m in 0..nm {
                            lfma(&mut acc, &adj[j][m], shat_of(m, 0, kappa)?);
                            if let Some(less) = sub(kappa, m) {
                                let factor = cscale(&adj[j][m], &(two.clone() * &self.prototype.from_usize(kappa[m] as usize)));
                                lfma(&mut acc, &csub(&zero, &factor), &get(&cur, (1, less))?);
                            }
                        }
                        if let Some(prev) = &previous {
                            lfma(&mut acc, &csub(&zero, &delta), &get(prev, (0, add(kappa, j)))?);
                        }
                        let out: L4<T> = std::array::from_fn(|slot| cmul(&acc[slot], &inv_adjf));
                        cur.insert((0, kappa), out);
                    }
                }
            }
            if let Some(prev) = &previous {
                // Largest relative change per rank, ranks <= target.
                let mut change = 0.0f64;
                for r in 1..=target {
                    let (mut diff, mut size) = (0.0f64, 0.0f64);
                    for (key, value) in cur.iter().filter(|(key, _)| rank_of(key) == r) {
                        let old = &prev[key];
                        for slot in 0..4 {
                            diff = diff.max(cnorm2(&csub(&value[slot], &old[slot])).to_f64().sqrt());
                            size = size.max(cnorm2(&value[slot]).to_f64().sqrt());
                        }
                    }
                    if size > 0.0 {
                        change = change.max(diff / size);
                    }
                }
                if !change.is_finite() || (t >= 2 && change > last_change) {
                    // Diverging: keep the previous pass.
                    break;
                }
                last_change = change;
                chosen = Some(cur.clone());
                if change <= f64::EPSILON {
                    break;
                }
            } else if passes == 0 {
                chosen = Some(cur.clone());
                last_change = 0.0;
            }
            previous = Some(cur);
        }
        // Alternative PV reduction (hep-ph/0509141 sect. 3.2, eqs. (PValtrec1),
        // (PValtrec2); COLLIER's CalcDpv2), the last resort when neither
        // expansion converges: the modified Cayley matrix X instead of Z, so
        // metric coefficients divide by Delta once per two ranks and momentum
        // coefficients by det X (Passarino-Veltman divides by the Gram matrix at
        // every rank). Exact up to rounding; triangles of rank target - 1. With
        // r_n = Shat_n(l, k') - 2 k'_n C(l+1, k'-e_n) and Q = 2l + |k'|:
        //   2 Delta ((Q+5-N) - 2 eps) C(l+1, k') = det X C(l, k') + Delta C0(l, k')
        //       - sum_n X~_0n r_n,
        //   det X C(0, k'+e_i) = X~_i0 [2 ((Q+5-N) - 2 eps) C(1, k') - C0(0, k')]
        //       + sum_n X~_in r_n.
        let alt_pv = || -> Result<HashMap<Key, L4<T>>, String> {
            let mut det_x = cmul(&cscale(&m02, &two), &delta);
            for a in 0..nm {
                for b in 0..nm {
                    det_x = csub(&det_x, &cmul(&cmul(&f[a], &f[b]), &adj[a][b]));
                }
            }
            if cnorm2(&det_x).is_zero() || cnorm2(&delta).is_zero() {
                return Err("alternative PV: singular modified Cayley or Gram matrix".into());
            }
            let inv_det_x = cinv(&det_x);
            let inv_two_delta = cinv(&cscale(&delta, &two));
            let xt0 = |n: usize| csub(&zero, &adjf[n]);
            let xtm: Vec<Vec<C<T>>> = (0..nm)
                .map(|i| {
                    (0..nm)
                        .map(|j| {
                            let mut v = cscale(&cmul(&m02, &adj[i][j]), &two);
                            for n1 in 0..nm {
                                for m1 in 0..nm {
                                    if let Some(w) = second_adjugate(&z, nm, i, n1, j, m1) {
                                        v = cadd(&v, &cmul(&w, &cmul(&f[n1], &f[m1])));
                                    }
                                }
                            }
                            v
                        })
                        .collect()
                })
                .collect();
            let mut cur: HashMap<Key, L4<T>> = HashMap::new();
            cur.insert((0, [0, 0, 0]), d0.clone());
            let get = |map: &HashMap<Key, L4<T>>, key: Key| -> Result<L4<T>, String> {
                map.get(&key).cloned().ok_or_else(|| format!("alternative PV: coefficient {key:?} missing"))
            };
            for q in 0..target {
                for lambda in 0..=q / 2 {
                    for kappa in compositions(q - 2 * lambda, nm) {
                        let mut rhs: Vec<L4<T>> = Vec::with_capacity(nm);
                        for n1 in 0..nm {
                            let mut value = shat_of(n1, lambda, kappa)?.clone();
                            if let Some(less) = sub(kappa, n1) {
                                lfma(&mut value, &csub(&zero, &real(2 * kappa[n1] as usize)), &get(&cur, (lambda + 1, less))?);
                            }
                            rhs.push(value);
                        }
                        let c0 = spz.get(&(lambda, kappa)).ok_or("alternative PV: C0 outside the triangle ranks")?;
                        let mut acc = lzero(&zero);
                        lfma(&mut acc, &det_x, &get(&cur, (lambda, kappa))?);
                        lfma(&mut acc, &delta, c0);
                        for n1 in 0..nm {
                            lfma(&mut acc, &csub(&zero, &xt0(n1)), &rhs[n1]);
                        }
                        let mut metric: L4<T> = std::array::from_fn(|slot| cmul(&acc[slot], &inv_two_delta));
                        let a = self.prototype.from_usize(q as usize + 4 - nm);
                        div_dimension(&mut metric, &(self.prototype.one() / &a));
                        cur.insert((lambda + 1, kappa), metric.clone());
                        if lambda == 0 {
                            mul_dimension(&mut metric, &a);
                            let mut bracket = lzero(&zero);
                            lfma(&mut bracket, &real(2), &metric);
                            lfma(&mut bracket, &csub(&zero, &real(1)), c0);
                            let first = (0..nm).find(|&i| kappa[i] > 0).unwrap_or(nm);
                            for i in 0..nm.min(first + 1) {
                                let mut acc = lzero(&zero);
                                lfma(&mut acc, &xt0(i), &bracket);
                                for n1 in 0..nm {
                                    lfma(&mut acc, &xtm[i][n1], &rhs[n1]);
                                }
                                let out: L4<T> = std::array::from_fn(|slot| cmul(&acc[slot], &inv_det_x));
                                cur.insert((0, add(kappa, i)), out);
                            }
                        }
                    }
                }
            }
            Ok(cur)
        };
        let chosen = match chosen {
            _ if self.gram_alt_only => {
                // COLLIER's a priori estimate of this reduction (CalcDred,
                // w_pv2): rounding grows like w^ceil(r/2), w = max |(Zadj f)_j|
                // / |Delta|; its rotated copy shares that error (Delta and
                // det X do not depend on the labelling), so large w is
                // refused here instead of trusted to the rotation test.
                let size = |z: &C<T>| cnorm2(z).to_f64().sqrt();
                let w = (0..nm).map(|j| size(&adjf[j])).fold(0.0f64, f64::max) / size(&delta);
                let estimate = 2f64.powi(-53) * w.max(1.0).powi((i32::from(target) + 1) / 2);
                if !(estimate <= ALTERNATIVE_PV_LIMIT) {
                    return Err(format!("alternative PV: estimated error {estimate:e}"));
                }
                alt_pv()?
            }
            Some(map) if last_change <= EXPANSION_TOLERANCE => map,
            _ => 'gc: {
                // Small (Zadj f)_j as well (Xtilde >> Zadj f): the expansion
                // for small Gram and small modified Cayley determinants
                // (hep-ph/0509141, eqs. (recurGramCayley1), (recurGramCayley2);
                // COLLIER's CalcDgy). Metric coefficients from
                //   2 sum_m k_m Zadj_km C(l+1, k-e_m) = sum_n Zadj_kn Shat_n(l, k)
                //       + Y_k0 C(l, k) - Delta C(l, k+e_k),
                // momentum coefficients from
                //   Xtilde_ij C(0, k) = Zadj_ij [(2(P+5-N) - 4 eps) C(1, k) - C0(0, k)]
                //       + sum_mn Zadj2_(in)(jm) f_n [Shat_m(0, k) - 2 k_m C(1, k-e_m)]
                //       + Y_0j C(0, k+e_i),
                // Y_0j = -(Zadj f)_j; each pass gains a factor Delta, Y_0j and
                // costs two ranks.
                let mut xt = [[zero.clone(), zero.clone(), zero.clone()], [zero.clone(), zero.clone(), zero.clone()], [zero.clone(), zero.clone(), zero.clone()]];
                let (mut gi, mut gj) = (0, 0);
                for a in 0..nm {
                    for b in 0..nm {
                        let mut value = cscale(&cmul(&m02, &adj[a][b]), &two);
                        for n1 in 0..nm {
                            for m1 in 0..nm {
                                if let Some(w) = second_adjugate(&z, nm, a, n1, b, m1) {
                                    value = cadd(&value, &cmul(&w, &cmul(&f[n1], &f[m1])));
                                }
                            }
                        }
                        xt[a][b] = value;
                        if cnorm2(&xt[a][b]) > cnorm2(&xt[gi][gj]) {
                            (gi, gj) = (a, b);
                        }
                    }
                }
                if cnorm2(&xt[gi][gj]).is_zero() || child_rank < target + 1 {
                    break 'gc alt_pv()?;
                }
                let inv_xt = cinv(&xt[gi][gj]);
                let y0 = |j: usize| csub(&zero, &adjf[j]);
                let gc_passes = (child_rank - 1 - target) / 2;
                let mut previous: Option<HashMap<Key, L4<T>>> = None;
                let mut gc_change = f64::INFINITY;
                let mut gc_chosen: Option<HashMap<Key, L4<T>>> = None;
                let mut momentum_rank = child_rank + 1;
                for t in 0..=gc_passes {
                    let metric_rank = if t == 0 { child_rank + 1 } else { momentum_rank };
                    momentum_rank = metric_rank - 2;
                    let mut cur: HashMap<Key, L4<T>> = HashMap::new();
                    cur.insert((0, [0, 0, 0]), d0.clone());
                    let get = |map: &HashMap<Key, L4<T>>, key: Key| -> Result<L4<T>, String> {
                        map.get(&key).cloned().ok_or_else(|| format!("small-Gram/Cayley expansion: coefficient {key:?} missing"))
                    };
                    for r in 2..=metric_rank {
                        for level in 1..=r / 2 {
                            let mut kappas = compositions(r - 2 * level, nm);
                            kappas.sort_by_key(|kappa| std::cmp::Reverse(kappa[l]));
                            for target_kappa in kappas {
                                let lambda = level - 1;
                                let kappa = add(target_kappa, l);
                                let mut acc = lzero(&zero);
                                for m in 0..nm {
                                    lfma(&mut acc, &adj[k][m], shat_of(m, lambda, kappa)?);
                                }
                                if let Some(prev) = &previous {
                                    lfma(&mut acc, &y0(k), &get(prev, (lambda, kappa))?);
                                    lfma(&mut acc, &csub(&zero, &delta), &get(prev, (lambda, add(kappa, k)))?);
                                }
                                for m in (0..nm).filter(|&m| m != l) {
                                    if let Some(less) = sub(kappa, m) {
                                        let factor = cscale(&adj[k][m], &(two.clone() * &self.prototype.from_usize(kappa[m] as usize)));
                                        lfma(&mut acc, &csub(&zero, &factor), &get(&cur, (level, less))?);
                                    }
                                }
                                let divisor = cinv(&cscale(&adj[k][l], &(two.clone() * &self.prototype.from_usize(kappa[l] as usize))));
                                let out: L4<T> = std::array::from_fn(|slot| cmul(&acc[slot], &divisor));
                                cur.insert((level, target_kappa), out);
                            }
                        }
                    }
                    for p in 1..=momentum_rank {
                        for kappa in compositions(p, nm) {
                            let mut metric = get(&cur, (1, kappa))?;
                            mul_dimension(&mut metric, &self.prototype.from_usize(p as usize + 4 - nm));
                            let mut bracket = lzero(&zero);
                            lfma(&mut bracket, &real(2), &metric);
                            lfma(&mut bracket, &csub(&zero, &real(1)), spz.get(&(0, kappa)).ok_or("small-Gram/Cayley expansion: C0 outside the triangle ranks")?);
                            let mut acc = lzero(&zero);
                            lfma(&mut acc, &adj[gi][gj], &bracket);
                            for n1 in 0..nm {
                                for m1 in 0..nm {
                                    let Some(w) = second_adjugate(&z, nm, gi, n1, gj, m1) else { continue };
                                    let wf = cmul(&w, &f[n1]);
                                    lfma(&mut acc, &wf, shat_of(m1, 0, kappa)?);
                                    if let Some(less) = sub(kappa, m1) {
                                        let factor = cscale(&wf, &(two.clone() * &self.prototype.from_usize(kappa[m1] as usize)));
                                        lfma(&mut acc, &csub(&zero, &factor), &get(&cur, (1, less))?);
                                    }
                                }
                            }
                            if let Some(prev) = &previous {
                                lfma(&mut acc, &y0(gj), &get(prev, (0, add(kappa, gi)))?);
                            }
                            let out: L4<T> = std::array::from_fn(|slot| cmul(&acc[slot], &inv_xt));
                            cur.insert((0, kappa), out);
                        }
                    }
                    if let Some(prev) = &previous {
                        let mut change = 0.0f64;
                        for r in 1..=target {
                            let (mut diff, mut size) = (0.0f64, 0.0f64);
                            for (key, value) in cur.iter().filter(|(key, _)| rank_of(key) == r) {
                                let Some(old) = prev.get(key) else { continue };
                                for slot in 0..4 {
                                    diff = diff.max(cnorm2(&csub(&value[slot], &old[slot])).to_f64().sqrt());
                                    size = size.max(cnorm2(&value[slot]).to_f64().sqrt());
                                }
                            }
                            if size > 0.0 {
                                change = change.max(diff / size);
                            }
                        }
                        if !change.is_finite() || (t >= 2 && change > gc_change) {
                            break;
                        }
                        gc_change = change;
                        gc_chosen = Some(cur.clone());
                        if change <= f64::EPSILON {
                            break;
                        }
                    }
                    previous = Some(cur);
                }
                match gc_chosen {
                    Some(map) if gc_change <= EXPANSION_TOLERANCE => map,
                    _ => alt_pv().map_err(|error| format!(
                            "small-Gram expansion did not converge (last relative change {last_change:e}; with small Cayley determinant {gc_change:e}); {error}"
                        ))?,
                }
            }
        };
        // Write the coefficients of rank <= target in the layout order.
        let layout = lookup(mask).ok_or("small-Gram expansion: no box layout")?;
        for (&(lambda, ref kappa), &index) in layout {
            let key: Key = (lambda, std::array::from_fn(|i| kappa.get(i).copied().unwrap_or(0)));
            self.values[index as usize] = if rank_of(&key) <= target {
                chosen.get(&key).cloned().ok_or("small-Gram expansion: coefficient missing")?
            } else {
                lzero(&zero)
            };
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_adjugate_satisfies_the_paper_identities() {
        // Eq. (Zadjadjrelation0): sum_m Zadj2_(im)(jl) Z_mk = Zadj_il d_jk - Zadj_ij d_lk.
        // N = 2: the paper's d_il d_kj - d_ij d_kl with Zadj = [[z11, -z01], [-z10, z00]].
        let e2 = [[1.3, -0.4], [-0.4, 0.8]];
        let z2: [[C<f64>; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| C::new(if i < 2 && j < 2 { e2[i][j] } else { 0.0 }, 0.0))
        });
        let adj2 = |i: usize, j: usize| match (i, j) {
            (0, 0) => e2[1][1],
            (1, 1) => e2[0][0],
            _ => -e2[j][i],
        };
        for i in 0..2 {
            for j in 0..2 {
                for l in 0..2 {
                    for k in 0..2 {
                        let mut lhs = 0.0;
                        for m in 0..2 {
                            if let Some(w) = second_adjugate(&z2, 2, i, m, j, l) {
                                lhs += w.re * e2[m][k];
                            }
                        }
                        let rhs = adj2(i, l) * f64::from(u8::from(j == k)) - adj2(i, j) * f64::from(u8::from(l == k));
                        assert!((lhs - rhs).abs() < 1e-12, "N=2 i{i} j{j} l{l} k{k}: {lhs} vs {rhs}");
                    }
                }
            }
        }
        let entries = [[2.0, 0.3, -1.1], [0.3, -0.7, 0.45], [-1.1, 0.45, 1.9]];
        let z: [[C<f64>; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| C::new(entries[i][j], 0.0)));
        let adj = |i: usize, j: usize| {
            let (a, b, c, d) = ((j + 1) % 3, (j + 2) % 3, (i + 1) % 3, (i + 2) % 3);
            entries[a][c] * entries[b][d] - entries[a][d] * entries[b][c]
        };
        for i in 0..3 {
            for j in 0..3 {
                for l in 0..3 {
                    for k in 0..3 {
                        let mut lhs = 0.0;
                        for m in 0..3 {
                            if let Some(w) = second_adjugate(&z, 3, i, m, j, l) {
                                lhs += w.re * entries[m][k];
                            }
                        }
                        let rhs = adj(i, l) * f64::from(u8::from(j == k)) - adj(i, j) * f64::from(u8::from(l == k));
                        assert!((lhs - rhs).abs() < 1e-12, "i{i} j{j} l{l} k{k}: {lhs} vs {rhs}");
                    }
                }
            }
        }
    }
}
