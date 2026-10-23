//! The packed Cartesian tensor basis used by the IREGI MadLoop interface.

/// Exponents of `(q0, q1, q2, q3)` for one tensor moment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Monomial(pub [u16; 4]);

impl Monomial {
    pub fn rank(self) -> usize {
        self.0.iter().map(|&x| usize::from(x)).sum()
    }
}

/// Increasing rank, then decreasing time, x and y exponents.
///
/// This is an index enumeration, not a symbolic polynomial representation.
pub fn tensor_basis(max_rank: u16) -> Vec<Monomial> {
    let mut basis = Vec::new();
    for rank in 0..=max_rank {
        for t in (0..=rank).rev() {
            for x in (0..=rank - t).rev() {
                for y in (0..=rank - t - x).rev() {
                    basis.push(Monomial([t, x, y, rank - t - x - y]));
                }
            }
        }
    }
    basis
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_rank_two_order() {
        assert_eq!(
            tensor_basis(2)[5..],
            [
                Monomial([2, 0, 0, 0]),
                Monomial([1, 1, 0, 0]),
                Monomial([1, 0, 1, 0]),
                Monomial([1, 0, 0, 1]),
                Monomial([0, 2, 0, 0]),
                Monomial([0, 1, 1, 0]),
                Monomial([0, 1, 0, 1]),
                Monomial([0, 0, 2, 0]),
                Monomial([0, 0, 1, 1]),
                Monomial([0, 0, 0, 2]),
            ]
        );
        assert_eq!(tensor_basis(7).len(), 330);
    }
}
