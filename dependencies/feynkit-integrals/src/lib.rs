//! FeynKit's exact affine propagator partial fractions without graph types.
//!
//! The partial-fraction engine and exact Gaussian-rational polynomial/matrix
//! helpers are extracted from FeynKit revision `865afdc43fb4e6b950439d34b4ec2c38352873e7`.
//! See `UPSTREAM.md` for provenance and the deliberately narrower constructor.
//! Symbolica performs every polynomial manipulation, rank calculation and solve.

use std::collections::BTreeSet;

use symbolica::{
    atom::{Atom, AtomCore, AtomView},
    domains::{
        algebraic::AlgebraicExtension,
        rational::{Q, RationalField},
        rational_polynomial::{RationalPolynomial, RationalPolynomialField},
    },
    poly::{PolyVariable, polynomial::MultivariatePolynomial},
    tensors::matrix::Matrix,
};
use thiserror::Error;

// Keep the copied engine byte-identical apart from its documented type alias.
#[rustfmt::skip]
mod partial_fraction;

type GaussianField = AlgebraicExtension<RationalField>;
type FamilyCoefficient = RationalPolynomial<GaussianField, u16>;
type FamilyField = RationalPolynomialField<GaussianField, u16>;
type CoefficientMatrix = Matrix<FamilyField>;
type ExactPolynomial = MultivariatePolynomial<FamilyField, u16>;

/// The affine subset of FeynKit's integral-family errors.
#[derive(Clone, Debug, Error)]
pub enum IntegralFamilyError {
    #[error("supply a nonempty list of distinct bare Symbolica symbols as affine variables")]
    InvalidVariables,
    #[error("invalid affine propagator basis: {0}")]
    InvalidBasis(String),
    #[error("supply one signed integer power per denominator")]
    InvalidPowers,
    #[error("partial fractioning exceeded its {0}-state budget")]
    PartialFractionLimit(usize),
    #[error("a propagator power overflowed during partial fractioning")]
    PowerOverflow,
}

pub use IntegralFamilyError as AffineFamilyError;

/// Ordered inverse propagators affine in explicit independent symbols.
///
/// All dependence on the integrated variables must use the supplied symbols.
/// Other atoms are external coefficients. In particular, the constructor has
/// no knowledge of momentum names, scalar-product identities or on-shell rules.
/// Coefficients are exact rational functions over Gaussian rationals; floating
/// literals are rejected by Symbolica's exact conversion.
///
/// Specialize exact kinematic relations before construction. A decomposition
/// obtained for generic symbolic coefficients need not remain valid where one
/// of its coefficient denominators vanishes.
#[derive(Clone, Debug)]
pub struct AffineFamily {
    // Preserve the upstream field name to keep the engine/helper extraction
    // unchanged. These are caller-supplied affine symbols, not graph objects.
    scalar_products: Vec<Atom>,
    denominators: Vec<Atom>,
    rank: usize,
}

impl AffineFamily {
    /// Validate the complete ordered family, including currently unused rows.
    /// Bare, distinct symbols avoid ambiguity between a variable such as
    /// `f(k)` and a different expression `g(k)` containing the same momentum.
    pub fn new(variables: Vec<Atom>, denominators: Vec<Atom>) -> Result<Self, IntegralFamilyError> {
        if variables.is_empty()
            || variables
                .iter()
                .any(|v| !matches!(v.as_view(), AtomView::Var(_)))
            || variables.iter().collect::<BTreeSet<_>>().len() != variables.len()
        {
            return Err(IntegralFamilyError::InvalidVariables);
        }
        let mut family = Self {
            scalar_products: variables,
            denominators,
            rank: 0,
        };
        // Retain the upstream eager check of both linear and constant
        // coefficients, including dependence hidden inside opaque functions.
        family.rank = family.rank_of(&family.denominators)?;
        Ok(family)
    }

    pub fn variables(&self) -> &[Atom] {
        &self.scalar_products
    }

    pub fn denominators(&self) -> &[Atom] {
        &self.denominators
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn is_independent(&self) -> bool {
        self.rank == self.denominators.len()
    }

    // This is the upstream external-coefficient guard specialized to the
    // explicit-symbol API: graph momentum aliases do not exist here.
    fn require_external_coefficient(&self, coefficient: &Atom) -> Result<(), IntegralFamilyError> {
        if self
            .scalar_products
            .iter()
            .any(|p| coefficient.contains(p.as_view()))
        {
            return Err(IntegralFamilyError::InvalidBasis(format!(
                "coefficient depends on affine variables: {coefficient}"
            )));
        }
        Ok(())
    }
}

include!("affine_helpers.rs");
