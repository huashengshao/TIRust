//! Exact guards for derivative neighborhoods of the scalar-master formulas.
//!
//! These conservative chart guards exclude selected Gram, threshold, and
//! modified-Cayley degenerations. Their nonvanishing is not a proof that every
//! formula branch supports arbitrary Dual directions; qualification tests and
//! the evaluator's other input/output checks remain necessary. In particular,
//! an individual zero external invariant is not rejected by itself.

use oneloop::ScalarIntegral;
use symbolica::{
    atom::{Atom, Symbol},
    domains::atom::AtomField,
    tensors::matrix::Matrix,
};

/// One when every selected discriminant is nonzero, and zero otherwise.
///
/// Nested native conditionals avoid underflow from multiplying discriminants
/// with different kinematic dimensions. They also retain lazy evaluation.
pub(crate) fn domain_discriminant(family: ScalarIntegral, p: &[Atom]) -> Atom {
    let guards = match family {
        ScalarIntegral::A0 => return Atom::num(1),
        ScalarIntegral::B0 | ScalarIntegral::DB0 => {
            vec![bubble_discriminant(p)]
        }
        ScalarIntegral::C0 => {
            let masses = &p[3..6];
            let edges = [(0, 1, 0), (1, 2, 1), (0, 2, 2)];
            let mut guards = vec![kallen(&p[2], &p[0], &p[1])];
            guards.extend(
                edges
                    .iter()
                    .map(|&(i, j, invariant)| kallen(&p[invariant], &masses[i], &masses[j])),
            );
            guards.push(exact_determinant(cayley_matrix(masses, p, &edges)));
            guards
        }
        ScalarIntegral::D0 => {
            let masses = &p[6..10];
            let edges = [
                (0, 1, 0),
                (1, 2, 1),
                (2, 3, 2),
                (0, 3, 3),
                (0, 2, 4),
                (1, 3, 5),
            ];
            let dot12 = &p[4] - &p[0] - &p[1];
            let dot23 = &p[5] - &p[1] - &p[2];
            let dot13 = &p[3] + &p[1] - &p[4] - &p[5];
            let mut guards = vec![exact_determinant(vec![
                vec![Atom::num(2) * &p[0], dot12.clone(), dot13.clone()],
                vec![dot12, Atom::num(2) * &p[1], dot23.clone()],
                vec![dot13, dot23, Atom::num(2) * &p[2]],
            ])];
            guards.extend(
                edges
                    .iter()
                    .map(|&(i, j, invariant)| kallen(&p[invariant], &masses[i], &masses[j])),
            );
            let cayley = cayley_matrix(masses, p, &edges);
            for omitted in 0..4 {
                let indices: Vec<_> = (0..4).filter(|&index| index != omitted).collect();
                let minor = indices
                    .iter()
                    .map(|&i| indices.iter().map(|&j| cayley[i][j].clone()).collect())
                    .collect();
                guards.push(exact_determinant(minor));
            }
            guards.push(exact_determinant(cayley));
            guards.extend(
                [(0, 1, 4), (1, 2, 5), (2, 3, 4), (3, 0, 5)]
                    .map(|(a, b, c)| kallen(&p[c], &p[a], &p[b])),
            );
            guards
        }
    };
    guards.into_iter().rev().fold(Atom::num(1), |next, guard| {
        Symbol::IF.call((guard, next, Atom::num(0)))
    })
}

/// Dimensionless bubble discriminant, dividing before any squared values.
/// The first nonzero mass is the pivot; the invariant handles massless bubbles.
/// Its zero/nonzero classification is invariant under a common nonzero scale.
pub(crate) fn bubble_discriminant(p: &[Atom]) -> Atom {
    let scale = Symbol::IF.call((
        &p[1],
        &p[1],
        Symbol::IF.call((&p[2], &p[2], Symbol::IF.call((&p[0], &p[0], 1)))),
    ));
    kallen(&(&p[0] / &scale), &(&p[1] / &scale), &(&p[2] / &scale))
}

fn kallen(invariant: &Atom, first: &Atom, second: &Atom) -> Atom {
    let difference = invariant - first - second;
    &difference * &difference - Atom::num(4) * first * second
}

/// Modified Cayley matrix: Y_ii = 2 m_i² and
/// Y_ij = m_i² + m_j² - (q_i - q_j)².
fn cayley_matrix(
    masses: &[Atom],
    invariants: &[Atom],
    edges: &[(usize, usize, usize)],
) -> Vec<Vec<Atom>> {
    let mut matrix = vec![vec![Atom::num(0); masses.len()]; masses.len()];
    for (i, mass) in masses.iter().enumerate() {
        matrix[i][i] = Atom::num(2) * mass;
    }
    for &(i, j, invariant) in edges {
        let entry = &masses[i] + &masses[j] - &invariants[invariant];
        matrix[i][j] = entry.clone();
        matrix[j][i] = entry;
    }
    matrix
}

// Reuse audit: Symbolica's Matrix::det supports AtomField and is already used
// by the master boundary guard and FeynKit tensor reduction. Constructing the
// matrix here leaves all determinant algebra to Symbolica. Disable statistical
// zero testing so preparation never treats an inconclusive result as zero.
fn exact_determinant(matrix: Vec<Vec<Atom>>) -> Atom {
    let field = AtomField {
        statistical_zero_test: false,
        ..AtomField::new()
    };
    Matrix::from_nested_vec(matrix, field)
        .expect("fixed square derivative-domain matrix")
        .det()
        .expect("fixed square derivative-domain matrix")
}
