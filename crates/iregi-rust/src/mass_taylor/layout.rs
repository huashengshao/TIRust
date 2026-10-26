//! Private source proposal only: auxiliary mass/user Taylor layout metadata.
//!
//! No Taylor arithmetic is implemented here. Rectangle enumeration/order is
//! adapted from gammalooprs::utils::hyperdual_utils::mixed_derivative_shape
//! (references/gammaloop_feynkit/crates/gammalooprs/src/utils/hyperdual_utils.rs),
//! which is currently pub(crate). Shape preflight follows the existing
//! iregi_masters::dual_validation checks before Symbolica's panicking dynamic
//! constructor. Symbolica HyperDual exclusively owns multiplication metadata.
//! See source-reuse-notes.md for the three reuse checks and restrictions.

use std::collections::HashSet;

use crate::ReductionError;
use symbolica::domains::dual::HyperDual;

/// Explicit caller resource limits, not mathematical propagator-power caps.
/// No Default is supplied: the private owner must choose its finite work budget.
/// These bound logical layout coordinates and candidate metadata pair visits;
/// Symbolica exposes no fallible dynamic constructor or strict byte estimator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DottedLayoutBudget {
    pub max_joint_components: usize,
    pub max_shape_entries: usize,
    /// Total candidate pair visits across our HyperDual construction and the
    /// validator's second canonical construction: 2*C*(C-1).
    pub max_metadata_pair_visits: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DottedLayoutResources {
    pub auxiliary_components: usize,
    pub user_components: usize,
    pub joint_components: usize,
    pub joint_dimensions: usize,
    pub shape_entries: usize,
    pub metadata_pair_visits: usize,
    pub analytic_jet_order: u16,
}

/// Internal auxiliary-major product B_aux x S_user, preserving S_user's order.
/// A selected coefficient is a Taylor coefficient: no extra factorial applies
/// when the internal plan starts with unit powers.
#[derive(Clone, Debug)]
pub(crate) struct DottedLayout {
    resources: DottedLayoutResources,
    user_shape: Option<Vec<Vec<usize>>>,
    joint_shape: Vec<Vec<usize>>,
    user_components: usize,
    embedded_user_indices: Vec<usize>,
    selected_indices: Vec<usize>,
    mass_seeds: Vec<(usize, usize)>,
}

fn invalid(message: &str) -> ReductionError {
    ReductionError::InvalidInput(format!("mass-Taylor layout: {message}"))
}

/// Validate only shape/index metadata. No multiplication table is reproduced.
/// Unlike HyperDual::new, this returns errors for malformed external shapes.
fn validate_shape(shape: &[Vec<usize>]) -> Result<(), ReductionError> {
    let primal = shape.first().ok_or_else(|| invalid("empty user Dual shape"))?;
    if primal.iter().any(|&degree| degree != 0) {
        return Err(invalid("the user primal must be first"));
    }
    let axes = primal.len();
    let mut seen = HashSet::with_capacity(shape.len());
    for component in shape {
        if component.len() != axes {
            return Err(invalid("user multi-index dimensions differ"));
        }
        if !seen.insert(component.as_slice()) {
            return Err(invalid("duplicate user multi-index"));
        }
        let mut total = 0usize;
        for &degree in component {
            total = total.checked_add(degree)
                .ok_or_else(|| invalid("user component degree overflows usize"))?;
            degree.checked_add(degree)
                .ok_or_else(|| invalid("user degree addition overflows usize"))?;
        }
        if total >= shape.len() {
            return Err(invalid("user component degree exceeds its ancestor count"));
        }
    }
    for component in shape {
        let mut predecessor = component.clone();
        for axis in 0..axes {
            if predecessor[axis] != 0 {
                predecessor[axis] -= 1;
                if !seen.contains(predecessor.as_slice()) {
                    return Err(invalid("user shape is not ancestor-closed"));
                }
                predecessor[axis] += 1;
            }
        }
    }
    Ok(())
}

fn checked_product(left: usize, right: usize) -> Result<usize, ReductionError> {
    left.checked_mul(right).ok_or_else(|| invalid("component count overflows usize"))
}

/// Match Workspace::new_dual's existing analytic jet-order type before
/// asking Symbolica to allocate the joint multiplication table. This is an
/// engine order bound, not an additional component-count or fixed-array limit.
fn validate_analytic_jet_order(shape: &[Vec<usize>]) -> Result<(), ReductionError> {
    shape.iter().try_fold(0u16, |maximum, component| {
        component.iter().try_fold(0usize, |total, &degree| total.checked_add(degree))
            .and_then(|degree| u16::try_from(degree).ok())
            .map(|degree| maximum.max(degree))
            .ok_or_else(|| invalid("analytic joint derivative order exceeds u16"))
    })?;
    Ok(())
}

/// Preflight cardinalities/work without enumerating auxiliary components,
/// allocating a validation HashSet, or constructing any HyperDual metadata.
fn preflight(
    powers: &[u16], users: &[Vec<usize>], budget: DottedLayoutBudget,
) -> Result<DottedLayoutResources, ReductionError> {
    if powers.is_empty() || powers.iter().any(|&power|power==0) {
        return Err(invalid("a nonempty positive propagator power vector are required"));
    }
    if powers.iter().all(|&power|power==1) {
        return Err(invalid("at least one nonunit propagator power is required"));
    }
    let primal=users.first().ok_or_else(||invalid("empty user Dual shape"))?;
    let auxiliary_components=powers.iter().try_fold(1usize,|count,&power|
        checked_product(count,usize::from(power)))?;
    let user_components=users.len();
    let joint_components=checked_product(auxiliary_components,user_components)?;
    if joint_components>budget.max_joint_components {
        return Err(ReductionError::Preparation("mass-Taylor layout layout resource budget: joint components exceeded".into()));
    }
    let active_axes=powers.iter().filter(|&&power|power>1).count();
    let joint_dimensions=active_axes.checked_add(primal.len())
        .ok_or_else(||invalid("joint multi-index dimension overflows usize"))?;
    let shape_entries=checked_product(joint_components,joint_dimensions)?;
    if shape_entries>budget.max_shape_entries {
        return Err(ReductionError::Preparation("mass-Taylor layout layout resource budget: shape entries exceeded".into()));
    }
    // Source inspection: HyperDual::from_values loops C*(C-1) candidates;
    // validate_dual_structure builds a second canonical HyperDual. This is a
    // metadata/control-work bound, never a table implementation or RSS claim.
    let per_constructor=checked_product(joint_components,joint_components-1)?;
    let metadata_pair_visits=checked_product(2,per_constructor)?;
    if metadata_pair_visits>budget.max_metadata_pair_visits {
        return Err(ReductionError::Preparation("mass-Taylor layout layout resource budget: Symbolica metadata pair visits exceeded".into()));
    }
    let auxiliary_order=powers.iter().try_fold(0usize,|sum,&power|
        sum.checked_add(usize::from(power-1)).ok_or_else(||invalid("total extra power overflows usize")))?;
    let mut user_order=0usize;
    for component in users {
        if component.len()!=primal.len(){return Err(invalid("user multi-index dimensions differ"));}
        let total=component.iter().try_fold(0usize,|sum,&degree|
            sum.checked_add(degree).ok_or_else(||invalid("user component degree overflows usize")))?;
        user_order=user_order.max(total);
    }
    let analytic_order=auxiliary_order.checked_add(user_order)
        .ok_or_else(||invalid("analytic joint derivative order overflows usize"))?;
    let analytic_jet_order=u16::try_from(analytic_order)
        .map_err(|_|invalid("analytic joint derivative order exceeds u16"))?;
    Ok(DottedLayoutResources{auxiliary_components,user_components,joint_components,
        joint_dimensions,shape_entries,metadata_pair_visits,analytic_jet_order})
}

/// Metadata-only adaptation of the existing FeynKit/gammaloop rectangle helper.
/// Cardinality/order/resource preflight must run before calling this enumerator.
fn auxiliary_rectangle(orders: &[usize]) -> Vec<Vec<usize>> {
    let mut shape = vec![Vec::new()];
    for &maximum in orders {
        let previous = std::mem::take(&mut shape);
        for prefix in previous {
            for degree in 0..=maximum {
                let mut component = prefix.clone();
                component.push(degree);
                shape.push(component);
            }
        }
    }
    // Same order as mixed_derivative_shape: leading primal, single axes in
    // axis order, then higher degrees lexicographically. User order is separate.
    shape.sort_by(|left, right| {
        let left_degree: usize = left.iter().sum();
        let right_degree: usize = right.iter().sum();
        left_degree.cmp(&right_degree).then_with(|| {
            if left_degree == 1 { right.cmp(left) } else { left.cmp(right) }
        })
    });
    shape
}

impl DottedLayout {
    /// Positive powers, at least one nonunit, with explicit caller limits.
    /// None represents an ordinary caller, with the internal user shape `[[]]`.
    /// Some preserves the complete ordered user shape, including inactive axes.
    pub(crate) fn new(
        powers: &[u16],
        user_shape: Option<&[Vec<usize>]>,
        budget: DottedLayoutBudget,
    ) -> Result<Self, ReductionError> {
        let scalar_shape = [Vec::new()];
        let users = user_shape.unwrap_or(&scalar_shape);
        let resources=preflight(powers,users,budget)?;
        validate_shape(users)?;
        let user_components = resources.user_components;
        let active: Vec<_> = powers.iter().enumerate().filter_map(|(line, &power)| {
            (power > 1).then_some((line, usize::from(power - 1)))
        }).collect();
        let dimensions = resources.joint_dimensions;
        let auxiliary_orders: Vec<_> = active.iter().map(|&(_, order)| order).collect();
        let auxiliary = auxiliary_rectangle(&auxiliary_orders);
        if auxiliary.len()!=resources.auxiliary_components {
            return Err(invalid("auxiliary rectangle cardinality differs from preflight"));
        }
        let joint_components = resources.joint_components;
        let mut joint_shape = Vec::with_capacity(joint_components);
        for auxiliary_component in &auxiliary {
            for user_component in users {
                let mut joint = Vec::with_capacity(dimensions);
                joint.extend_from_slice(auxiliary_component);
                joint.extend_from_slice(user_component);
                joint_shape.push(joint);
            }
        }
        validate_shape(&joint_shape)?;
        validate_analytic_jet_order(&joint_shape)?;
        // Canonical validation/table construction belongs to Symbolica, never
        // to this helper. Values are irrelevant for this metadata-only carrier.
        let canonical = HyperDual::<u8>::new(joint_shape.clone());
        iregi_masters::validate_dual_structure(&canonical)
            .map_err(|error| invalid(&format!("invalid joint Symbolica shape: {error}")))?;

        let selected_auxiliary = auxiliary.iter().position(|component| component == &auxiliary_orders)
            .ok_or_else(|| invalid("target auxiliary coefficient is absent"))?;
        let selected_base = checked_product(selected_auxiliary, user_components)?;
        let embedded_user_indices = (0..user_components).collect();
        let selected_indices = (0..user_components).map(|user|selected_base.checked_add(user)
            .ok_or_else(||invalid("selected coefficient index overflows usize"))).collect::<Result<Vec<_>,_>>()?;
        let mut mass_seeds = Vec::with_capacity(active.len());
        for (axis, &(line, _)) in active.iter().enumerate() {
            let mut linear = vec![0; active.len()];
            linear[axis] = 1;
            let auxiliary_index = auxiliary.iter().position(|component| component == &linear)
                .ok_or_else(|| invalid("auxiliary linear mass seed is absent"))?;
            mass_seeds.push((line, checked_product(auxiliary_index, user_components)?));
        }
        Ok(Self {
            resources,
            user_shape: user_shape.map(<[Vec<usize>]>::to_vec),
            joint_shape,
            user_components,
            embedded_user_indices,
            selected_indices,
            mass_seeds,
        })
    }

    pub(crate) fn resources(&self) -> DottedLayoutResources { self.resources }
    pub(crate) fn joint_shape(&self) -> &[Vec<usize>] { &self.joint_shape }
    pub(crate) fn user_shape(&self) -> Option<&[Vec<usize>]> { self.user_shape.as_deref() }
    pub(crate) fn user_components(&self) -> usize { self.user_components }
    pub(crate) fn joint_components(&self) -> usize { self.joint_shape.len() }
    pub(crate) fn embedded_user_indices(&self) -> &[usize] { &self.embedded_user_indices }
    pub(crate) fn selected_indices(&self) -> &[usize] { &self.selected_indices }
    pub(crate) fn mass_seeds(&self) -> &[(usize, usize)] { &self.mass_seeds }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbolica::{
        atom::{Atom, AtomCore, Symbol},
        domains::{
            dual::{DualNumber, DualNumberStructure},
            float::Complex,
            integer::Integer,
            rational::{Q, Rational},
        },
        evaluate::{Dualizer, ExpressionEvaluator},
    };

    symbolica::create_hyperdual_from_components!(ReorderedSparseUser,
        [[0, 0], [0, 1], [1, 0], [2, 0], [1, 1]]);

    fn budget() -> DottedLayoutBudget {
        DottedLayoutBudget{max_joint_components:512,max_shape_entries:8192,
            max_metadata_pair_visits:2*512*511}
    }
    fn layout(powers:&[u16],users:Option<&[Vec<usize>]>) -> Result<DottedLayout,ReductionError> {
        DottedLayout::new(powers,users,budget())
    }

    fn variables() -> [Atom; 3] {
        [symbolica::parse!("dotted_layout_test::q"),
         symbolica::parse!("dotted_layout_test::m0"),
         symbolica::parse!("dotted_layout_test::m1")]
    }

    fn rational_roots(variables: &[Atom; 3], powers: &[u16]) -> Atom {
        let [q, m0, m1] = variables;
        let first = q.pow(2) - m0;
        let second = (q + Atom::num(1)).pow(2) - m1;
        first.pow(-i64::from(powers[0])) * second.pow(-i64::from(powers[1]))
    }

    fn evaluate_joint(layout: &DottedLayout, input: &[Rational]) -> Vec<Rational> {
        let parameters = variables();
        let root = rational_roots(&parameters, &[1; 6]);
        let exact: ExpressionEvaluator<Complex<Rational>> = root.evaluator(&parameters)
            .direct_translation(true).horner_iterations(0).build().unwrap();
        let mut evaluator = exact.vectorize(&Dualizer::new(
            HyperDual::<Complex<Rational>>::new(layout.joint_shape().to_vec()), vec![],
        )).unwrap().map_to_ring(&Q).unwrap();
        let mut output = vec![Rational::from(0); layout.joint_components()];
        evaluator.evaluate_in_ring(input, &mut output, &Q);
        output
    }

    // Test input transport only. Neither this function nor the layout helper
    // implements Taylor products: all coefficient arithmetic is Symbolica's.
    fn lift(layout: &DottedLayout, inputs: &[Vec<Rational>]) -> Vec<Rational> {
        assert_eq!(inputs.len(), 3);
        let mut result = vec![Rational::from(0); inputs.len() * layout.joint_components()];
        for (argument, user_values) in inputs.iter().enumerate() {
            assert_eq!(user_values.len(), layout.user_components());
            for (&joint, value) in layout.embedded_user_indices().iter().zip(user_values) {
                result[argument * layout.joint_components() + joint] = value.clone();
            }
        }
        for &(line, joint) in layout.mass_seeds() {
            assert!(line < 2, "this test expression only has masses0/1");
            result[(line + 1) * layout.joint_components() + joint] = Rational::from(1);
        }
        result
    }

    /// Independent exact oracle: Symbolica differentiates explicit dotted
    /// powers on a nonlinear user path, then evaluates at the origin. It never
    /// uses the auxiliary layout, selected indices or its Taylor program.
    fn user_coefficients(expression: &Atom, shape: &[Vec<usize>], user_axes: &[Symbol]) -> Vec<Rational> {
        shape.iter().map(|component| {
            let mut derivative = expression.clone();
            for (&axis, &order) in user_axes.iter().zip(component) {
                for _ in 0..order { derivative = derivative.derivative(axis); }
                derivative = derivative / Atom::num(Integer::factorial(order as u32));
            }
            for &axis in user_axes { derivative = derivative.replace(axis).with(Atom::num(0)); }
            Rational::try_from(&derivative.expand()).unwrap()
        }).collect()
    }

    #[test]
    fn auxiliary_indices_preserve_requested_lines_and_sparse_user_order() {
        let shape = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![2, 0], vec![1, 1]];
        let layout = layout(&[2, 1, 1, 1, 1, 2], Some(&shape)).unwrap();
        assert_eq!(layout.user_shape(), Some(shape.as_slice()));
        assert_eq!(layout.user_components(), 5);
        assert_eq!(layout.joint_components(), 20);
        for (user, component) in shape.iter().enumerate() {
            let embedded = &layout.joint_shape()[layout.embedded_user_indices()[user]];
            assert_eq!(&embedded[..2], &[0, 0]);
            assert_eq!(&embedded[2..], component);
            let selected = &layout.joint_shape()[layout.selected_indices()[user]];
            assert_eq!(&selected[..2], &[1, 1]);
            assert_eq!(&selected[2..], component);
        }
        assert_eq!(layout.mass_seeds().iter().map(|&(line, _)| line).collect::<Vec<_>>(), [0, 5]);
        for (axis, &(_, index)) in layout.mass_seeds().iter().enumerate() {
            let mut expected = vec![0; 4]; expected[axis] = 1;
            assert_eq!(layout.joint_shape()[index], expected);
        }
        let scalar = self::layout(&[1, 1, 1, 1, 1, 3], None).unwrap();
        assert_eq!(scalar.user_shape(), None);
        assert_eq!(scalar.user_components(), 1);
        assert_eq!(scalar.joint_shape()[scalar.selected_indices()[0]], [2]);
        assert_eq!(scalar.mass_seeds(), [(5, 1)]);
    }

    #[test]
    fn exact_one_dot_triple_and_mixed_coefficients_have_no_extra_factorial() {
        for (powers, expected) in [
            ([2, 1, 1, 1, 1, 1], Rational::from((1, 896))),
            ([3, 1, 1, 1, 1, 1], Rational::from((1, 7168))),
            ([4, 1, 1, 1, 1, 1], Rational::from((1, 57344))),
            ([2, 2, 1, 1, 1, 1], Rational::from((1, 12544))),
        ] {
            let layout = layout(&powers, None).unwrap();
            let input = lift(&layout, &[vec![Rational::from(3)], vec![Rational::from(1)], vec![Rational::from(2)]]);
            let before = input.clone();
            let actual = evaluate_joint(&layout, &input);
            assert_eq!(actual[layout.selected_indices()[0]], expected);
            assert_eq!(input, before);
        }
    }

    #[test]
    fn selected_user_jets_match_explicit_dots_on_nonlinear_mass_and_geometry_path() {
        let u = symbolica::symbol!("dotted_layout_test::u");
        let v = symbolica::symbol!("dotted_layout_test::v");
        let paths = [
            symbolica::parse!("3+dotted_layout_test::u+2*dotted_layout_test::v+dotted_layout_test::u^2+dotted_layout_test::u*dotted_layout_test::v"),
            symbolica::parse!("1+2*dotted_layout_test::u+dotted_layout_test::u^2+3*dotted_layout_test::u*dotted_layout_test::v"),
            symbolica::parse!("2+dotted_layout_test::v+dotted_layout_test::u*dotted_layout_test::v"),
        ];
        // Missing (0,2) is intentional: sparse ancestor-closed user shapes must
        // retain their supplied order, rather than becoming full rectangles.
        let shape = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![2, 0], vec![1, 1]];
        let user_input: Vec<_> = paths.iter().map(|path| user_coefficients(path, &shape, &[u, v])).collect();
        for powers in [[2, 1, 1, 1, 1, 1], [3, 1, 1, 1, 1, 1], [4, 1, 1, 1, 1, 1], [2, 2, 1, 1, 1, 1], [3, 2, 1, 1, 1, 1]] {
            let layout = layout(&powers, Some(&shape)).unwrap();
            let actual = evaluate_joint(&layout, &lift(&layout, &user_input));
            let selected: Vec<_> = layout.selected_indices().iter().map(|&index| actual[index].clone()).collect();
            let expected = user_coefficients(&rational_roots(&paths, &powers), &shape, &[u, v]);
            assert_eq!(selected, expected, "powers={powers:?}");
            assert_ne!(expected[4], Rational::from(0), "mixed user component must be exercised");
            assert_ne!(expected[3], Rational::from(0), "second user coefficient must be exercised");
        }
    }

    #[test]
    fn static_and_dynamic_user_carriers_share_layout_and_preserve_all_selected_values() {
        let mut static_user = ReorderedSparseUser { values: [
            Rational::from(3), Rational::from(2), Rational::from(1), Rational::from(1), Rational::from(4),
        ] };
        let shape: Vec<_> = static_user.get_shape().into_iter().map(<[usize]>::to_vec).collect();
        let dynamic_user = HyperDual::from_values(shape.clone(), static_user.values().to_vec());
        iregi_masters::validate_dual_structure(&static_user).unwrap();
        iregi_masters::validate_dual_structure(&dynamic_user).unwrap();
        let a = layout(&[3, 1, 1, 1, 1, 1], Some(&shape)).unwrap();
        let dynamic_shape: Vec<_> = dynamic_user.get_shape().into_iter().map(<[usize]>::to_vec).collect();
        let b = layout(&[3, 1, 1, 1, 1, 1], Some(&dynamic_shape)).unwrap();
        assert_eq!(a.joint_shape(), b.joint_shape());
        assert_eq!(a.selected_indices(), b.selected_indices());
        let constants = |value| vec![Rational::from(value), Rational::from(0), Rational::from(0), Rational::from(0), Rational::from(0)];
        let actual_static = evaluate_joint(&a, &lift(&a, &[static_user.values().to_vec(), constants(1), constants(2)]));
        let actual_dynamic = evaluate_joint(&b, &lift(&b, &[dynamic_user.values().to_vec(), constants(1), constants(2)]));
        assert_eq!(actual_static, actual_dynamic);
        let selected: Vec<_> = a.selected_indices().iter().map(|&index| actual_static[index].clone()).collect();
        static_user.values_mut().clone_from_slice(&selected);
        assert_eq!(static_user.get_shape(), shape.iter().map(Vec::as_slice).collect::<Vec<_>>());
        assert_eq!(static_user.values(), selected);
    }

    #[test]
    fn invalid_power_shape_and_overflow_inputs_return_errors_before_symbolica_construction() {
        for powers in [vec![1; 5], vec![1; 7], vec![0, 1, 1, 1, 1, 1], vec![1; 6]] {
            assert!(layout(&powers, None).is_err());
        }
        for shape in [vec![], vec![vec![1]], vec![vec![0], vec![0]],
            vec![vec![0, 0], vec![1]], vec![vec![0], vec![2]],
            vec![vec![0, 0], vec![1, 1], vec![0, 1]]] {
            assert!(layout(&[2, 1, 1, 1, 1, 1], Some(&shape)).is_err(), "shape={shape:?}");
        }
        let overflow = vec![vec![0], vec![usize::MAX]];
        let error = layout(&[2, 1, 1, 1, 1, 1], Some(&overflow)).unwrap_err();
        assert!(error.to_string().contains("overflows usize"));
        assert!(checked_product(usize::MAX, 2).is_err());
        assert_eq!(checked_product(4, 5).unwrap(), 20);
        // Exercise the engine-order preflight on exact component labels only.
        // These are not passed to HyperDual: a genuinely ancestor-closed shape
        // beyond the bound would require a large, unnecessary allocation.
        let maximum = usize::from(u16::MAX);
        assert!(validate_analytic_jet_order(&[vec![maximum - 1, 1]]).is_ok());
        let unsupported = validate_analytic_jet_order(&[vec![maximum, 1]]).unwrap_err();
        assert!(unsupported.to_string().contains("analytic joint derivative order exceeds u16"));
        assert!(validate_analytic_jet_order(&[vec![usize::MAX, 1]]).is_err());
        let constant_axes = vec![vec![0, 0]];
        let valid = layout(&[2, 1, 1, 1, 1, 1], Some(&constant_axes)).unwrap();
        assert_eq!(valid.joint_shape(), [vec![0, 0, 0], vec![1, 0, 0]]);
    }

    #[test]
    fn five_point_higher_last_power_and_explicit_work_budgets_are_checked() {
        let users=vec![vec![0,0],vec![0,1],vec![1,0]];
        let actual=layout(&[2,1,1,1,1],Some(&users)).unwrap();
        assert_eq!(actual.resources().joint_components,6);
        assert_eq!(actual.resources().shape_entries,18);
        assert_eq!(actual.resources().metadata_pair_visits,60);
        let last=layout(&[1,1,1,1,5],None).unwrap();
        assert_eq!(last.joint_components(),5);
        assert_eq!(last.mass_seeds(),[(4,1)]);
        assert_eq!(last.joint_shape()[last.selected_indices()[0]],[4]);
        let too_small=DottedLayoutBudget{max_joint_components:1,..budget()};
        assert!(DottedLayout::new(&[2,1,1,1,1],None,too_small).is_err());
        let too_small=DottedLayoutBudget{max_shape_entries:17,..budget()};
        assert!(DottedLayout::new(&[2,1,1,1,1],Some(&users),too_small).is_err());
        let too_small=DottedLayoutBudget{max_metadata_pair_visits:59,..budget()};
        assert!(DottedLayout::new(&[2,1,1,1,1],Some(&users),too_small).is_err());
        assert!(DottedLayout::new(&[u16::MAX;5],None,budget()).is_err());
    }

    #[test]
    fn five_point_unit_start_symbolica_coefficients_match_explicit_higher_dots() {
        for powers in [[2,1,1,1,1],[3,1,1,1,1],[4,1,1,1,1],[2,2,1,1,1]] {
            let actual_layout=layout(&powers,None).unwrap();
            let input=lift(&actual_layout,&[vec![Rational::from(3)],vec![Rational::from(1)],vec![Rational::from(2)]]);
            let evaluated=evaluate_joint(&actual_layout,&input);
            // The existing exact oracle in this module already differentiates
            // normalized user paths. Here compare the independently written
            // explicit-power rational expression at the same exact point.
            let parameters=variables();
            let oracle:ExpressionEvaluator<Complex<Rational>>=rational_roots(&parameters,&powers)
                .evaluator(&parameters).direct_translation(true).build().unwrap();
            let values=[Rational::from(3),Rational::from(1),Rational::from(2)];
            let expected=oracle.map_to_ring(&Q).unwrap().evaluate_single_in_ring(&values,&Q);
            assert_eq!(evaluated[actual_layout.selected_indices()[0]],expected);
        }
    }

}
