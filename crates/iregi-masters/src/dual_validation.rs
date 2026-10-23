//! Validate caller-provided Dual metadata before Symbolica vectorization.
//!
//! Symbolica owns the canonical multiplication table. Its static shape helper,
//! dynamic `HyperDual::from_values` validation/tests, and FeynKit's dynamic Dual
//! adapters were checked: there is no fallible dynamic metadata validator.
//! The checks here validate indices and dimensions, then compare against
//! Symbolica's table; they do not implement Dual arithmetic.

use std::collections::HashSet;

use symbolica::domains::dual::{DualNumberStructure, HyperDual};

use crate::MasterError;

/// Reject malformed shapes and inconsistent metadata before building evaluators.
///
/// The shape must contain a unique leading primal, unique equally sized
/// multi-indices, and every immediate ancestor. The supplied multiplication
/// table and maximum total degree must agree with Symbolica's dynamic carrier.
/// Equivalent multiplication-table orderings are accepted. Validation itself
/// does not initialize Symbolica's symbolic state or require its runtime license.
pub fn validate_dual_structure<D: DualNumberStructure + ?Sized>(
    dual: &D,
) -> Result<(), MasterError> {
    let shape = dual.get_shape();
    if shape.is_empty() || shape.len() != dual.get_len() {
        return Err(invalid("empty or inconsistent shape"));
    }
    let variables = shape[0].len();
    if shape[0].iter().any(|&degree| degree != 0) {
        return Err(invalid("the primal must be first"));
    }
    let mut seen = HashSet::with_capacity(shape.len());
    for component in &shape {
        if component.len() != variables {
            return Err(invalid(
                "all component multi-indices must have the same length",
            ));
        }
        if !seen.insert(*component) {
            return Err(invalid("component multi-indices must be unique"));
        }
        let mut total = 0usize;
        for &degree in *component {
            total = total
                .checked_add(degree)
                .ok_or_else(|| invalid("component total degree overflows usize"))?;
            // HyperDual's canonical builder adds pairs of component degrees.
            // Check before calling it, so hostile metadata cannot overflow.
            degree
                .checked_add(degree)
                .ok_or_else(|| invalid("component degree sum overflows usize"))?;
        }
        // An ancestor-closed degree-k shape needs at least k+1 components.
        // This also bounds work before constructing the canonical carrier.
        if total >= shape.len() {
            return Err(invalid(
                "component degree exceeds the available ancestor count",
            ));
        }
    }
    for component in &shape {
        let mut ancestor = component.to_vec();
        for index in 0..variables {
            if ancestor[index] != 0 {
                ancestor[index] -= 1;
                if !seen.contains(ancestor.as_slice()) {
                    return Err(invalid("shape is not ancestor-closed"));
                }
                ancestor[index] += 1;
            }
        }
    }
    let table = dual.get_multiplication_table();
    if table.iter().any(|&(left, right, output)| {
        left >= shape.len() || right >= shape.len() || output >= shape.len()
    }) {
        return Err(invalid("multiplication-table index is outside the shape"));
    }

    // All constructor preconditions and all degree additions are now checked.
    // u8 values are irrelevant: only Symbolica's shape metadata is inspected.
    let canonical =
        HyperDual::<u8>::new(shape.iter().map(|component| component.to_vec()).collect());
    if dual.get_max_depth() != canonical.get_max_depth() {
        return Err(invalid("maximum derivative depth disagrees with the shape"));
    }
    let mut supplied = table.to_vec();
    let mut expected = canonical.get_multiplication_table().to_vec();
    supplied.sort_unstable();
    expected.sort_unstable();
    if supplied != expected {
        return Err(invalid(
            "multiplication table disagrees with Symbolica's shape metadata",
        ));
    }
    Ok(())
}

fn invalid(reason: &str) -> MasterError {
    MasterError::InvalidShape(reason.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Metadata {
        shape: Vec<Vec<usize>>,
        len: usize,
        table: Vec<(usize, usize, usize)>,
        depth: usize,
    }

    impl Metadata {
        fn from_shape(shape: Vec<Vec<usize>>) -> Self {
            let canonical = HyperDual::<u8>::new(shape.clone());
            Self {
                len: shape.len(),
                table: canonical.get_multiplication_table().to_vec(),
                depth: canonical.get_max_depth(),
                shape,
            }
        }
    }

    impl DualNumberStructure for Metadata {
        fn get_len(&self) -> usize {
            self.len
        }
        fn get_shape(&self) -> Vec<&[usize]> {
            self.shape.iter().map(Vec::as_slice).collect()
        }
        fn get_multiplication_table(&self) -> &[(usize, usize, usize)] {
            &self.table
        }
        fn get_max_depth(&self) -> usize {
            self.depth
        }
    }

    fn rejects(metadata: &impl DualNumberStructure, reason: &str) {
        let error = validate_dual_structure(metadata).unwrap_err();
        assert!(matches!(error, MasterError::InvalidShape(_)));
        assert!(error.to_string().contains(reason), "{error}");
    }

    #[test]
    fn rejects_duplicate_indices_even_in_a_symbolica_dynamic_carrier() {
        let duplicate = HyperDual::<u8>::new(vec![vec![0], vec![1], vec![1]]);
        rejects(&duplicate, "unique");
    }

    #[test]
    fn rejects_missing_ancestors_before_calling_the_panicking_constructor() {
        let mut metadata = Metadata::from_shape(vec![vec![0, 0], vec![1, 0], vec![0, 1]]);
        metadata.shape[2] = vec![1, 1];
        rejects(&metadata, "ancestor-closed");
    }

    #[test]
    fn rejects_inconsistent_dimensions_and_overflowing_degrees() {
        let mut metadata = Metadata::from_shape(vec![vec![0], vec![1]]);
        metadata.len = 3;
        rejects(&metadata, "inconsistent");
        metadata.len = 2;
        metadata.shape[1] = vec![1, 0];
        rejects(&metadata, "same length");
        metadata.shape[1] = vec![usize::MAX];
        rejects(&metadata, "overflows");
        metadata.shape = vec![vec![0, 0, 0], vec![usize::MAX / 2; 3]];
        rejects(&metadata, "overflows");
    }

    #[test]
    fn rejects_incorrect_tables_and_underreported_depth() {
        let mut metadata = Metadata::from_shape(vec![vec![0], vec![1], vec![2]]);
        metadata.depth = 1;
        rejects(&metadata, "depth");
        metadata.depth = 2;
        metadata.table[0].2 = metadata.len;
        rejects(&metadata, "outside");
        metadata.table[0].2 = 0;
        rejects(&metadata, "disagrees");
        metadata = Metadata::from_shape(vec![vec![0], vec![1], vec![2]]);
        metadata.table.pop();
        rejects(&metadata, "disagrees");
    }

    symbolica::create_hyperdual_from_components!(
        MixedStatic,
        [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1]]
    );

    #[test]
    fn accepts_static_dynamic_and_reordered_canonical_tables() {
        let static_dual = MixedStatic { values: [0u8; 5] };
        validate_dual_structure(&static_dual).unwrap();
        let dynamic = HyperDual::<u8>::new(MixedStatic::<u8>::SHAPE.map(Vec::from).to_vec());
        validate_dual_structure(&dynamic).unwrap();
        let mut metadata =
            Metadata::from_shape(dynamic.get_shape().iter().map(|s| s.to_vec()).collect());
        metadata.table.reverse();
        validate_dual_structure(&metadata).unwrap();
        validate_dual_structure(&HyperDual::<u8>::new(vec![vec![]])).unwrap();
    }
}
