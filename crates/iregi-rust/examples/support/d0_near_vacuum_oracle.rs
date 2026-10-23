//! Independent exact parameter-integral oracles for finite massive boxes.
//!
//! This optional validation module calls neither reduction nor scalar masters.
//! Symbolica constructs and integrates the parameter series, differentiates its
//! exact mass dependence, and supplies the Taylor factorial normalization. The
//! reference expands only in eta/M; a candidate that expands all deviations
//! around a frozen auxiliary mass therefore has a different truncation.
//! The separate exact-vacuum oracle differentiates the unexpanded integrand
//! before integration, with arbitrary heterogeneous input jets.

use serde_json::json;
use std::{
    collections::HashSet,
    sync::{Arc, OnceLock},
    time::Instant,
};
use symbolica::{
    atom::{Atom, AtomCore, Symbol},
    domains::{
        atom::AtomField,
        float::Complex,
        integer::{Integer, Z},
        rational::{Q, Rational},
    },
    id::Replacement,
    poly::PolyVariable,
    tensors::matrix::Matrix,
};

/// Exact centers and absolute complex remainder bounds in the requested order.
pub struct NearVacuumOracle {
    pub centers: Vec<Complex<Rational>>,
    pub radii: Vec<Rational>,
    pub metadata: serde_json::Value,
}

// These are limits on this validation helper's symbolic preparation, not on the
// production reduction API. Reject before constructing any symbolic expression.
const MAX_SERIES_ORDER: usize = 8;
const MAX_TAYLOR_DEGREE: usize = 4;
const MAX_COMPONENTS: usize = 15;
const EQUAL_MASS_OFFSETS: [i64; 4] = [0, 0, 0, 0];
const UNEQUAL_MASS_OFFSETS: [i64; 4] = [-1, 0, 1, 2];

#[derive(Clone)]
struct IntegratedSeries {
    expression: Atom,
    parameter_terms: usize,
}

// Separate tables for the two fixed mass-offset patterns make the structural
// cache key (offset pattern, inclusive order). Numerical samples, precision,
// derivative shapes, centers, and error bounds are deliberately absent.
// OnceLock also serializes concurrent initialization of the same exact series.
static INTEGRATED_SERIES: [OnceLock<Result<IntegratedSeries, String>>; MAX_SERIES_ORDER + 1] =
    [const { OnceLock::new() }; MAX_SERIES_ORDER + 1];
static UNEQUAL_INTEGRATED_SERIES: [OnceLock<Result<IntegratedSeries, String>>;
    MAX_SERIES_ORDER + 1] = [const { OnceLock::new() }; MAX_SERIES_ORDER + 1];

fn exact_field() -> AtomField {
    AtomField {
        statistical_zero_test: false,
        cancel_check_on_division: false,
        custom_normalization: None,
    }
}

fn validate_shape(shape: &[Vec<usize>]) -> Result<(), String> {
    if shape.is_empty() || shape.len() > MAX_COMPONENTS {
        return Err(format!(
            "oracle shape must contain 1..={MAX_COMPONENTS} components"
        ));
    }
    if shape[0].as_slice() != [0, 0] {
        return Err("oracle shape must start with the two-coordinate primal [0,0]".into());
    }
    let mut seen = HashSet::with_capacity(shape.len());
    for component in shape {
        if component.len() != 2
            || component
                .iter()
                .try_fold(0usize, |sum, degree| sum.checked_add(*degree))
                .is_none_or(|degree| degree > MAX_TAYLOR_DEGREE)
        {
            return Err(format!(
                "oracle components need two coordinates and total degree <= {MAX_TAYLOR_DEGREE}"
            ));
        }
        if !seen.insert(component.as_slice()) {
            return Err("oracle shape contains a duplicate component".into());
        }
    }
    for component in shape {
        for axis in 0..2 {
            if component[axis] > 0 {
                let mut ancestor = component.clone();
                ancestor[axis] -= 1;
                if !seen.contains(ancestor.as_slice()) {
                    return Err(format!(
                        "oracle shape is not ancestor-closed: missing {ancestor:?}"
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Differentiate with Symbolica, then apply its native Taylor normalization.
fn taylor_derivative(expression: &Atom, symbols: [Symbol; 2], alpha: &[usize]) -> Atom {
    let mut coefficient = expression.clone();
    for (symbol, degree) in symbols.into_iter().zip(alpha) {
        for _ in 0..*degree {
            coefficient = coefficient.derivative(symbol);
        }
        coefficient /= Atom::num(Integer::factorial(*degree as u32));
    }
    coefficient
}

/// Let Symbolica convert and evaluate both exact rational-polynomial parts.
fn evaluate_exact(
    expression: &Atom,
    variables: &Arc<Vec<PolyVariable>>,
    point: &[Complex<Rational>],
) -> Result<Complex<Rational>, String> {
    let rational = expression
        .try_to_rational_polynomial::<_, _, u16>(&Q, &Z, Some(variables.clone()))
        .map_err(|error| format!("exact rational-polynomial conversion: {error}"))?;
    if rational.numerator.get_vars_ref() != variables.as_slice()
        || rational.denominator.get_vars_ref() != variables.as_slice()
        || point.len() != variables.len()
    {
        return Err("unexpected variables in oracle rational expression".into());
    }
    let map = |value: &Integer| Complex::new(Rational::from(value), Rational::from(0));
    let numerator = rational.numerator.evaluate(map, point);
    let denominator = rational.denominator.evaluate(map, point);
    if denominator.re == Rational::from(0) && denominator.im == Rational::from(0) {
        return Err("zero exact denominator in oracle evaluation".into());
    }
    Ok(numerator / denominator)
}

fn prepare_integrated_series(
    series_order: usize,
    mass_offsets: [i64; 4],
) -> Result<IntegratedSeries, String> {
    let preparation_started = Instant::now();
    let z_symbol = symbolica::symbol!("iregi_d0_near_vacuum_oracle::z");
    let z = Atom::var(z_symbol);
    let x: Vec<_> = (0..4)
        .map(|i| {
            Atom::var(symbolica::symbol!(format!(
                "iregi_d0_near_vacuum_oracle::x{i}"
            )))
        })
        .collect();
    let u_symbols: Vec<_> = (0..3)
        .map(|i| symbolica::symbol!(format!("iregi_d0_near_vacuum_oracle::u{i}")))
        .collect();
    let u: Vec<_> = u_symbols.iter().copied().map(Atom::var).collect();

    // Derive the Symanzik polynomial from the four unit routing vectors with
    // metric (+---). This is independent of the candidate's ten-argument box
    // expression and avoids importing its invariant-to-parameter mapping.
    let routing: Vec<[Atom; 4]> = (0..4)
        .map(|i| std::array::from_fn(|mu| Atom::num(i64::from(i == mu))))
        .collect();
    let metric = [1, -1, -1, -1];
    let mut p = Atom::num(0);
    for i in 0..4 {
        for j in i + 1..4 {
            let distance: Atom = (0..4)
                .map(|mu| Atom::num(metric[mu]) * (&routing[i][mu] - &routing[j][mu]).pow(2))
                .sum();
            p -= &x[i] * &x[j] * distance;
        }
    }
    let expected = Atom::num(2) * (&x[1] * &x[2] + &x[1] * &x[3] + &x[2] * &x[3]);
    if !(&p - expected).expand().is_zero() {
        return Err("routing-derived Symanzik polynomial failed its identity".into());
    }
    // P=(x1+x2+x3)^2-(x1^2+x2^2+x3^2), so 0<=P<=1 on the simplex.
    let spatial_sum: Atom = x[1..].iter().cloned().sum();
    let spatial_squares: Atom = x[1..].iter().map(|value| value.pow(2)).sum();
    if !(&p - spatial_sum.pow(2) + spatial_squares)
        .expand()
        .is_zero()
    {
        return Err("uniform parameter bound identity failed".into());
    }
    let first_moment_check = match mass_offsets {
        EQUAL_MASS_OFFSETS => Rational::from((-1, 10)),
        // -2*integral(P + sum_i c_i*x_i) = -1/10 - 2*(2/24).
        // This is a check only; all coefficients are integrated below.
        UNEQUAL_MASS_OFFSETS => Rational::from((-4, 15)),
        _ => return Err("unsupported independent-oracle mass-offset pattern".into()),
    };
    let mass_deviation: Atom = x
        .iter()
        .zip(mass_offsets)
        .map(|(coordinate, offset)| Atom::num(offset) * coordinate)
        .sum();
    let parameter_polynomial = p + mass_deviation;

    let mut remaining = Atom::num(1);
    let mut simplex = Vec::with_capacity(4);
    for coordinate in &u {
        simplex.push(&remaining * coordinate);
        remaining *= Atom::num(1) - coordinate;
    }
    simplex.push(remaining);
    let jacobian = Matrix::from_nested_vec(
        simplex[..3]
            .iter()
            .map(|coordinate| {
                u_symbols
                    .iter()
                    .map(|&symbol| coordinate.derivative(symbol))
                    .collect()
            })
            .collect(),
        exact_field(),
    )
    .map_err(|error| format!("oracle Jacobian matrix: {error:?}"))?
    .det()
    .map_err(|error| format!("oracle Jacobian determinant: {error:?}"))?;
    let expected_jacobian = (Atom::num(1) - &u[0]).pow(2) * (Atom::num(1) - &u[1]);
    if !(&jacobian - expected_jacobian).expand().is_zero() {
        return Err("stick-breaking Jacobian identity failed".into());
    }
    let substitutions: Vec<_> = x
        .iter()
        .zip(&simplex)
        .map(|(parameter, value)| Replacement::new(parameter.clone(), value.clone()))
        .collect();
    let p_cube = parameter_polynomial.replace_multiple(substitutions.iter());
    let series = (Atom::num(1) / (Atom::num(1) + &z * p_cube).pow(2))
        .series(z_symbol, 0, series_order as u32)
        .map_err(|error| format!("oracle parameter series: {error}"))?
        .to_atom();
    eprintln!(
        "independent oracle N{series_order}: series {:?}",
        preparation_started.elapsed()
    );
    let cube_variables = Arc::new(
        std::iter::once(z.clone())
            .chain(u)
            .map(PolyVariable::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("oracle integration variables: {error}"))?,
    );
    let mut integrated = (jacobian * series)
        .try_to_polynomial::<_, u16>(&Q, Some(cube_variables))
        .map_err(|error| format!("oracle parameter polynomial: {error}"))?;
    let expanded_parameter_terms = integrated.nterms();
    eprintln!(
        "independent oracle N{series_order}: Q polynomial {:?}, {expanded_parameter_terms} terms",
        preparation_started.elapsed()
    );
    for variable in 1..4 {
        let antiderivative = integrated.integrate(variable);
        integrated = &antiderivative.replace(variable, &Rational::from(1))
            - &antiderivative.replace(variable, &Rational::from(0));
        eprintln!(
            "independent oracle N{series_order}: integral{variable} {:?}, {} terms",
            preparation_started.elapsed(),
            integrated.nterms()
        );
    }
    if !(1..4).all(|variable| integrated.degree(variable) == 0)
        || integrated.get_constant() != Rational::from((1, 6))
        || (series_order >= 1 && integrated.coefficient(&[1, 0, 0, 0]) != Some(first_moment_check))
    {
        return Err("oracle integration failed volume/first-moment checks".into());
    }
    Ok(IntegratedSeries {
        expression: integrated.to_expression(),
        parameter_terms: expanded_parameter_terms,
    })
}

/// Build a bounded exact oracle for p=eta*[0,-2,-2,0,0,-2], equal squared masses M.
///
/// Coordinates are `(t_mass,t_eta)`: all four squared masses are `M+t_mass`
/// and every occurrence of eta is `eta+t_eta`. Each output is the coefficient
/// of `t_mass^alpha[0] * t_eta^alpha[1]`, already divided by `alpha!`, in native
/// Symbolica Dual ordering. A typical shape is `[00,10,01,11,20,02]`.
///
/// The reference series retains orders 0 through `series_order` inclusive;
/// callers normally request 8. Every retained component has its own exact
/// remainder bound, even if its order exceeds the eta-series order. The
/// initial validation helper permits series order <=8 and Taylor degree <=4.
/// Only the integrated structural series is cached once per order/process;
/// derivatives, exact evaluation, and component bounds are computed per call.
///
/// Domain: Re(M)>0, Im(M)<=0, eta real, and |eta|<L, where
/// L=max(|Re(M)|,|Im(M)|). These exact conditions imply the uniform convergence
/// used in the proof. No floating-point arithmetic or scalar-master evaluation
/// is used. Calling this function initializes Symbolica and needs its runtime.
pub fn build_oracle(
    mass: &Complex<Rational>,
    eta: &Rational,
    shape: &[Vec<usize>],
    series_order: usize,
) -> Result<NearVacuumOracle, String> {
    build_offset_oracle(mass, eta, shape, series_order, EQUAL_MASS_OFFSETS)
}

/// Independent bounded oracle with squared masses `M+c_i*eta`, c=[-1,0,1,2].
///
/// Routing invariants remain `eta*[0,-2,-2,0,0,-2]`. The native coordinates are
/// common `M+t_mass` and `eta+t_eta`, so the mass eta-jets are c_i and the
/// common-mass jets are all one. The parameter polynomial is
/// `F=M+eta*R`, with `R=P+sum_i c_i*x_i` and `|R|<=3` on the simplex.
///
/// Expansion is around M, independent of a candidate choosing the first
/// physical mass `M-eta` as its frozen auxiliary mass. Require all four real
/// mass parts positive, Im(M)<=0, and `3*abs(eta)<L`, where
/// `L=max(abs(Re(M)),abs(Im(M)))`. Every Taylor component has its own exact
/// bound from `H_N(3*(abs(eta)+t_eta)/(L-t_mass))/(6*(L-t_mass)^2)`.
/// This family's structural series uses a separate offset-pattern/order cache.
pub fn build_unequal_mass_oracle(
    mass: &Complex<Rational>,
    eta: &Rational,
    shape: &[Vec<usize>],
    series_order: usize,
) -> Result<NearVacuumOracle, String> {
    build_offset_oracle(mass, eta, shape, series_order, UNEQUAL_MASS_OFFSETS)
}

fn build_offset_oracle(
    mass: &Complex<Rational>,
    eta: &Rational,
    shape: &[Vec<usize>],
    series_order: usize,
    mass_offsets: [i64; 4],
) -> Result<NearVacuumOracle, String> {
    validate_shape(shape)?;
    if series_order > MAX_SERIES_ORDER {
        return Err(format!(
            "oracle series order exceeds validation limit {MAX_SERIES_ORDER}"
        ));
    }
    if mass.re <= Rational::from(0) || mass.im > Rational::from(0) {
        return Err("oracle requires Re(M)>0 and Im(M)<=0".into());
    }
    let unequal_masses = mass_offsets == UNEQUAL_MASS_OFFSETS;
    if mass_offsets != EQUAL_MASS_OFFSETS && !unequal_masses {
        return Err("unsupported independent-oracle mass-offset pattern".into());
    }
    for offset in mass_offsets {
        if &mass.re + &(eta * &Rational::from(offset)) <= Rational::from(0) {
            return Err(
                "oracle requires every physical squared mass to have positive real part".into(),
            );
        }
    }
    let lower_norm = mass.re.abs().max(mass.im.abs());
    let eta_abs = eta.abs();
    let parameter_bound = if unequal_masses { 3 } else { 1 };
    let bound_ratio_numerator = Rational::from(parameter_bound) * &eta_abs;
    if bound_ratio_numerator >= lower_norm {
        return Err(if unequal_masses {
            "unequal-mass oracle requires 3*|eta|<max(|Re(M)|,|Im(M)|)".into()
        } else {
            "oracle requires |eta|<max(|Re(M)|,|Im(M)|)".into()
        });
    }
    let entry = if unequal_masses {
        &UNEQUAL_INTEGRATED_SERIES[series_order]
    } else {
        &INTEGRATED_SERIES[series_order]
    };
    let series_cache_hit = entry.get().is_some();
    let prepared = entry
        .get_or_init(|| prepare_integrated_series(series_order, mass_offsets))
        .clone()?;
    let integrated_expression = prepared.expression;
    let expanded_parameter_terms = prepared.parameter_terms;
    eprintln!("independent oracle N{series_order}: structural series cache hit={series_cache_hit}");
    let z_symbol = symbolica::symbol!("iregi_d0_near_vacuum_oracle::z");
    let z = Atom::var(z_symbol);

    let m_symbol = symbolica::symbol!("iregi_d0_near_vacuum_oracle::m");
    let eta_symbol = symbolica::symbol!("iregi_d0_near_vacuum_oracle::eta");
    let m = Atom::var(m_symbol);
    let eta_atom = Atom::var(eta_symbol);
    let symbols = [m_symbol, eta_symbol];
    let variables = Arc::new(vec![
        PolyVariable::from(m_symbol),
        PolyVariable::from(eta_symbol),
    ]);
    let approximation = integrated_expression
        .replace_multiple([Replacement::new(z.clone(), &eta_atom / &m)].iter())
        / m.pow(2);

    // H_N(z)=sum_{k>N}(k+1)z^k. The positive form makes the derivative-tail
    // proof transparent, including N smaller than a requested eta degree.
    let generating = Atom::num(1) / (Atom::num(1) - &z).pow(2);
    let tail_closed = z.pow((series_order + 1) as i64)
        * (Atom::num(series_order + 1) / (Atom::num(1) - &z)
            + Atom::num(1) / (Atom::num(1) - &z).pow(2));
    let tail_from_series = &generating
        - generating
            .series(z_symbol, 0, series_order as u32)
            .map_err(|error| format!("oracle remainder series: {error}"))?
            .to_atom();
    if !(tail_from_series - &tail_closed).together().is_zero() {
        return Err("positive generating-function tail identity failed".into());
    }
    // Bound(m,e)=H_N(B*e/m)/(6m^2), where B=1 or 3 is the uniform parameter
    // bound. The jet generator is Bound(L-t_mass,|eta|+t_eta).
    // Its components equal (-1)^a/(a!b!) d_m^a d_e^b R at (L,|eta|).
    // Each omitted term is bounded because |M|>=L and |R|<=B; integration
    // contributes simplex volume 1/6. All products and derivatives remain in
    // Symbolica; there is no independently implemented polynomial/jet engine.
    let remainder = tail_closed.replace_multiple(
        [Replacement::new(
            z,
            Atom::num(parameter_bound) * &eta_atom / &m,
        )]
        .iter(),
    ) / (Atom::num(6) * m.pow(2));
    let point = [mass.clone(), Complex::new(eta.clone(), Rational::from(0))];
    let bound_point = [
        Complex::new(lower_norm.clone(), Rational::from(0)),
        Complex::new(eta_abs.clone(), Rational::from(0)),
    ];
    let mut centers = Vec::with_capacity(shape.len());
    let mut radii = Vec::with_capacity(shape.len());
    let mut component_metadata = Vec::with_capacity(shape.len());
    for alpha in shape {
        let component_started = Instant::now();
        let center_expression = taylor_derivative(&approximation, symbols, alpha);
        eprintln!(
            "independent oracle N{series_order} {alpha:?}: center derivative {:?}",
            component_started.elapsed()
        );
        let center = evaluate_exact(&center_expression, &variables, &point)?;
        eprintln!(
            "independent oracle N{series_order} {alpha:?}: center evaluation {:?}",
            component_started.elapsed()
        );
        let mut radius_expression = taylor_derivative(&remainder, symbols, alpha);
        if alpha[0] % 2 != 0 {
            radius_expression = -radius_expression;
        }
        eprintln!(
            "independent oracle N{series_order} {alpha:?}: tail derivative {:?}",
            component_started.elapsed()
        );
        let radius_value = evaluate_exact(&radius_expression, &variables, &bound_point)?;
        eprintln!(
            "independent oracle N{series_order} {alpha:?}: tail evaluation {:?}",
            component_started.elapsed()
        );
        if radius_value.im != Rational::from(0) || radius_value.re < Rational::from(0) {
            return Err(format!(
                "nonpositive-real remainder certificate for {alpha:?}"
            ));
        }
        let radius = radius_value.re;
        component_metadata.push(json!({
            "multi_index_mass_eta": alpha,
            "center_real_exact_rational": center.re.to_string(),
            "center_imaginary_exact_rational": center.im.to_string(),
            "absolute_complex_remainder_bound_exact_rational": radius.to_string(),
        }));
        centers.push(center);
        radii.push(radius);
    }
    let mut metadata = json!({
        "method": "Independent routing-derived Symanzik parameter series; exact Symbolica polynomial integration and normalized derivatives",
        "series_order_inclusive": series_order,
        "expansion_variable": "z=eta/M; mass dependence retained exactly before differentiation",
        "integrated_series_in_z": integrated_expression.to_string(),
        "expanded_parameter_polynomial_terms": expanded_parameter_terms,
        "integrated_series_cache": {"scope": "process-local exact structural series keyed only by inclusive order", "already_initialized_on_entry": series_cache_hit},
        "coordinates": ["common squared mass: M+t_mass", "invariant scale: eta+t_eta"],
        "shape": shape,
        "coefficient_convention": "[t_mass^a t_eta^b], equal to partial derivatives divided by a! b!",
        "squared_mass_real_exact_rational": mass.re.to_string(),
        "squared_mass_imaginary_exact_rational": mass.im.to_string(),
        "eta_exact_rational": eta.to_string(),
        "invariant_multipliers": [0, -2, -2, 0, 0, -2],
        "mass_modulus_lower_bound_exact_rational": lower_norm.to_string(),
        "rho_exact_rational": (&bound_ratio_numerator / &lower_norm).to_string(),
        "normalization": "D0=integral_Delta3 (M+eta*P)^(-2) dx, volume(Delta3)=1/6; all pole components zero",
        "parameter_bound": "P=2*(x1*x2+x1*x3+x2*x3)=(x1+x2+x3)^2-(x1^2+x2^2+x3^2), hence 0<=P<=1",
        "remainder_generator": "H_N((abs(eta)+t_eta)/(L-t_mass))/(6*(L-t_mass)^2)",
        "positive_tail": "H_N(z)=z^(N+1)*((N+1)/(1-z)+1/(1-z)^2)",
        "remainder_proof": "Coefficientwise absolute values of every omitted eta-series term and all normalized mass/eta derivatives are majorized by the positive generator; |M|>=L and 0<=P<=1 uniformly, then integrate with volume 1/6",
        "components": component_metadata,
        "resource_limits": {"series_order": MAX_SERIES_ORDER, "total_taylor_degree": MAX_TAYLOR_DEGREE, "components": MAX_COMPONENTS},
        "rounding": "Exact Complex<Rational> centers and Rational bounds; no floating-point evaluation",
    });
    if unequal_masses {
        let metadata = metadata
            .as_object_mut()
            .expect("oracle metadata is an object");
        metadata.remove("squared_mass_real_exact_rational");
        metadata.remove("squared_mass_imaginary_exact_rational");
        metadata.insert(
            "reference_squared_mass".into(),
            json!({
                "real_exact_rational": mass.re.to_string(),
                "imaginary_exact_rational": mass.im.to_string(),
            }),
        );
        metadata.insert(
            "physical_squared_masses".into(),
            json!(mass_offsets.iter().map(|offset| {
            json!({
                "real_exact_rational": (&mass.re + &(eta * &Rational::from(*offset))).to_string(),
                "imaginary_exact_rational": mass.im.to_string(),
            })
        }).collect::<Vec<_>>()),
        );
        metadata.insert("mass_offsets".into(), json!(mass_offsets));
        metadata.insert(
            "coordinates".into(),
            json!([
                "common mass offset: M+t_mass",
                "eta+t_eta in invariants and all mass offsets; m_i=M+t_mass+c_i*(eta+t_eta)",
            ]),
        );
        metadata.insert("integrated_series_cache".into(), json!({
            "scope": "process-local exact structural series keyed by offset pattern and inclusive order",
            "mass_offsets": mass_offsets,
            "already_initialized_on_entry": series_cache_hit,
        }));
        metadata.insert(
            "normalization".into(),
            json!(
                "D0=integral_Delta3 (M+eta*R)^(-2) dx, volume(Delta3)=1/6; all pole components zero"
            ),
        );
        metadata.insert("parameter_bound".into(), json!("R=P+sum_i c_i*x_i, c=[-1,0,1,2]; 0<=P<=1 and -1<=sum_i c_i*x_i<=2, hence -1<=R<=3 and |R|<=3"));
        metadata.insert(
            "remainder_generator".into(),
            json!("H_N(3*(abs(eta)+t_eta)/(L-t_mass))/(6*(L-t_mass)^2)"),
        );
        metadata.insert("remainder_proof".into(), json!("Coefficientwise absolute values of every omitted eta-series term and all normalized common-mass/eta derivatives are majorized by the positive generator, retaining the factor 3^k in each term; |M|>=L and |R|<=3 uniformly, then integrate with volume 1/6"));
    }
    Ok(NearVacuumOracle {
        centers,
        radii,
        metadata,
    })
}

/// Exact finite Taylor jets at zero primal invariants and a common primal mass.
///
/// `physical_components` contains ten arguments, each with `shape.len()` Taylor
/// components: invariants `(s01,s12,s23,s03,s02,s13)`, then squared masses
/// `(m0,m1,m2,m3)`. All primal invariants must vanish and all primal squared
/// masses must equal `mass`. Nonprimal components may differ between every
/// argument and may include higher or mixed jets; the coordinates are generic
/// `(t0,t1)`, rather than the common-mass/eta coordinates of [`build_oracle`].
///
/// Symbolica differentiates the exact integrand `F(t,x)^(-2)` and sets `t=0`
/// before polynomial integration. Since `F(0,x)=M` on the simplex, every
/// requested derivative is an exactly integrable polynomial in the parameters
/// with powers of the nonzero constant M in its denominator. No expansion in
/// a bookkeeping parameter is used, so every returned radius is exactly zero.
/// Missing input monomials cannot affect a requested ancestor-closed component.
/// The same two-coordinate, degree-four resource limits apply as above.
pub fn build_vacuum_jet_oracle(
    mass: &Complex<Rational>,
    shape: &[Vec<usize>],
    physical_components: &[Complex<Rational>],
) -> Result<NearVacuumOracle, String> {
    let preparation_started = Instant::now();
    validate_shape(shape)?;
    let components = shape.len();
    if physical_components.len() != 10 * components {
        return Err("vacuum oracle requires ten argument-major sets of Taylor components".into());
    }
    if mass.re <= Rational::from(0) || mass.im > Rational::from(0) {
        return Err("vacuum oracle requires Re(M)>0 and Im(M)<=0".into());
    }
    let zero = Rational::from(0);
    for argument in 0..6 {
        let values = &physical_components[argument * components..(argument + 1) * components];
        if values[0].re != zero || values.iter().any(|value| value.im != zero) {
            return Err(
                "vacuum oracle requires zero primal invariants and real invariant jets".into(),
            );
        }
    }
    for argument in 6..10 {
        let values = &physical_components[argument * components..(argument + 1) * components];
        if values[0] != *mass {
            return Err("vacuum oracle requires equal primal squared masses".into());
        }
        if mass.im == zero && values[1..].iter().any(|value| value.im != zero) {
            return Err("zero-width vacuum masses require real mass jets".into());
        }
    }

    let m_symbol = symbolica::symbol!("iregi_d0_vacuum_jet_oracle::m");
    let imaginary_symbol = symbolica::symbol!("iregi_d0_vacuum_jet_oracle::imaginary_unit");
    let m = Atom::var(m_symbol);
    let imaginary = Atom::var(imaginary_symbol);
    let jet_symbols = [
        symbolica::symbol!("iregi_d0_vacuum_jet_oracle::t0"),
        symbolica::symbol!("iregi_d0_vacuum_jet_oracle::t1"),
    ];
    let jet_variables = jet_symbols.map(Atom::var);
    let x: Vec<_> = (0..4)
        .map(|i| {
            Atom::var(symbolica::symbol!(format!(
                "iregi_d0_vacuum_jet_oracle::x{i}"
            )))
        })
        .collect();
    // A formal imaginary-unit parameter keeps every Symbolica polynomial over
    // exact rational coefficients. Only its final exact Complex<Rational>
    // evaluation sets it to i, so complex jet seeds undergo no float conversion.
    let jet_deviation = |argument: usize| -> Atom {
        shape
            .iter()
            .enumerate()
            .skip(1)
            .map(|(component, alpha)| {
                let coefficient = &physical_components[argument * components + component];
                (Atom::num(coefficient.re.clone()) + Atom::num(coefficient.im.clone()) * &imaginary)
                    * jet_variables[0].pow(alpha[0] as i64)
                    * jet_variables[1].pow(alpha[1] as i64)
            })
            .sum()
    };
    let mut f = m.clone();
    // On sum(x_i)=1 the equal primal mass term is exactly M. Only nonprimal
    // mass jets enter this sum; all four physical derivatives are retained.
    for propagator in 0..4 {
        f += jet_deviation(6 + propagator) * &x[propagator];
    }
    for (argument, (left, right)) in [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 3)]
        .into_iter()
        .enumerate()
    {
        f -= jet_deviation(argument) * &x[left] * &x[right];
    }
    let integrand = Atom::num(1) / f.pow(2);
    let zero_jets: Vec<_> = jet_variables
        .iter()
        .map(|variable| Replacement::new(variable.clone(), Atom::num(0)))
        .collect();

    let cube_symbols: Vec<_> = (0..3)
        .map(|i| symbolica::symbol!(format!("iregi_d0_vacuum_jet_oracle::u{i}")))
        .collect();
    let cube: Vec<_> = cube_symbols.iter().copied().map(Atom::var).collect();
    let mut remaining = Atom::num(1);
    let mut simplex = Vec::with_capacity(4);
    for coordinate in &cube {
        simplex.push(&remaining * coordinate);
        remaining *= Atom::num(1) - coordinate;
    }
    simplex.push(remaining);
    if !(simplex.iter().cloned().sum::<Atom>() - Atom::num(1))
        .expand()
        .is_zero()
    {
        return Err("vacuum oracle simplex normalization failed".into());
    }
    let jacobian = Matrix::from_nested_vec(
        simplex[..3]
            .iter()
            .map(|coordinate| {
                cube_symbols
                    .iter()
                    .map(|&symbol| coordinate.derivative(symbol))
                    .collect()
            })
            .collect(),
        exact_field(),
    )
    .map_err(|error| format!("vacuum oracle Jacobian matrix: {error:?}"))?
    .det()
    .map_err(|error| format!("vacuum oracle Jacobian determinant: {error:?}"))?;
    let simplex_substitution: Vec<_> = x
        .iter()
        .zip(simplex)
        .map(|(parameter, coordinate)| Replacement::new(parameter.clone(), coordinate))
        .collect();
    let cube_variables = Arc::new(
        cube.iter()
            .cloned()
            .map(PolyVariable::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("vacuum oracle integration variables: {error}"))?,
    );
    let coefficient_variables = Arc::new(vec![
        PolyVariable::from(m_symbol),
        PolyVariable::from(imaginary_symbol),
    ]);
    let coefficient_point = [
        mass.clone(),
        Complex::new(Rational::from(0), Rational::from(1)),
    ];
    // Symbolica's integer numerator/denominator conversion, followed by its
    // coefficient map to Q, keeps all algebra in polynomial rings. The common
    // denominator may depend on M but cannot depend on a simplex coordinate;
    // verify this before pulling it outside the exact definite integrations.
    let mut rational_variables = cube_variables.as_ref().clone();
    rational_variables.extend(coefficient_variables.iter().cloned());
    let rational_variables = Arc::new(rational_variables);
    let mut centers = Vec::with_capacity(components);
    let mut component_metadata = Vec::with_capacity(components);
    for alpha in shape {
        let derivative =
            taylor_derivative(&integrand, jet_symbols, alpha).replace_multiple(zero_jets.iter());
        if jet_symbols
            .iter()
            .any(|&symbol| derivative.contains_symbol(symbol))
        {
            return Err("vacuum oracle derivative retains jet coordinates".into());
        }
        eprintln!(
            "exact vacuum oracle {alpha:?}: derivative {:?}",
            preparation_started.elapsed()
        );
        let parameter_integrand =
            &jacobian * derivative.replace_multiple(simplex_substitution.iter());
        let rational = parameter_integrand
            .try_to_rational_polynomial::<_, _, u16>(&Q, &Z, Some(rational_variables.clone()))
            .map_err(|error| format!("vacuum oracle rational parameter conversion: {error}"))?;
        if rational.numerator.get_vars_ref() != rational_variables.as_slice()
            || rational.denominator.get_vars_ref() != rational_variables.as_slice()
            || (0..3).any(|coordinate| rational.denominator.degree(coordinate) != 0)
        {
            return Err("vacuum oracle denominator depends on integration coordinates or unexpected variables".into());
        }
        let denominator = rational.denominator.to_expression();
        let mut polynomial = rational
            .numerator
            .map_coeff(|value| Rational::from(value), Q);
        if polynomial.nterms() > 100_000 {
            return Err("vacuum oracle parameter polynomial exceeds term limit".into());
        }
        let parameter_terms = polynomial.nterms();
        eprintln!(
            "exact vacuum oracle {alpha:?}: Q numerator {:?}, {parameter_terms} terms",
            preparation_started.elapsed()
        );
        for coordinate in 0..3 {
            let antiderivative = polynomial.integrate(coordinate);
            polynomial = &antiderivative.replace(coordinate, &Rational::from(1))
                - &antiderivative.replace(coordinate, &Rational::from(0));
        }
        if (0..3).any(|coordinate| polynomial.degree(coordinate) != 0) {
            return Err("vacuum oracle integration retains parameter coordinates".into());
        }
        let integrated = polynomial.to_expression() / denominator;
        let center = evaluate_exact(&integrated, &coefficient_variables, &coefficient_point)?;
        eprintln!(
            "exact vacuum oracle {alpha:?}: integrated and evaluated {:?}",
            preparation_started.elapsed()
        );
        component_metadata.push(json!({
            "multi_index": alpha,
            "parameter_polynomial_terms": parameter_terms,
            "integrated_expression": integrated.to_string(),
            "center_real_exact_rational": center.re.to_string(),
            "center_imaginary_exact_rational": center.im.to_string(),
            "absolute_complex_remainder_bound_exact_rational": "0",
        }));
        centers.push(center);
    }
    // A direct exact vacuum identity checks the measure and normalization.
    let vacuum_expected = evaluate_exact(
        &(Atom::num(1) / (Atom::num(6) * m.pow(2))),
        &coefficient_variables,
        &coefficient_point,
    )?;
    if centers[0] != vacuum_expected {
        return Err("vacuum oracle primal differs from 1/(6*M^2)".into());
    }
    let exact_inputs: Vec<_> = physical_components
        .chunks_exact(components)
        .map(|argument| {
            argument
                .iter()
                .map(|value| json!({"real": value.re.to_string(), "imaginary": value.im.to_string()}))
                .collect::<Vec<_>>()
        })
        .collect();
    let metadata = json!({
        "method": "Exact Symbolica differentiation of the unexpanded parameter integrand, jet-origin substitution, then polynomial integration",
        "series_truncation": "none: each requested finite Taylor coefficient is exact",
        "shape": shape,
        "coordinates": ["t0", "t1"],
        "argument_order": ["s01", "s12", "s23", "s03", "s02", "s13", "m0_squared", "m1_squared", "m2_squared", "m3_squared"],
        "physical_components_argument_major_exact_rational": exact_inputs,
        "coefficient_convention": "[t0^a t1^b], equal to partial derivatives divided by a! b!; input components already use the same normalization",
        "normalization": "D0=integral_Delta3 F^-2 dx, volume(Delta3)=1/6; all pole components zero",
        "exactness_proof": "At the jet origin F=M on the simplex. Differentiation followed by origin substitution leaves polynomial parameter numerators divided by powers of constant M. Symbolica integrates these polynomials exactly; ancestor closure ensures omitted input monomials cannot affect retained components.",
        "components": component_metadata,
        "resource_limits": {"total_taylor_degree": MAX_TAYLOR_DEGREE, "components": MAX_COMPONENTS, "parameter_polynomial_terms": 100_000},
        "rounding": "Exact Complex<Rational> values; no floating-point arithmetic",
    });
    Ok(NearVacuumOracle {
        centers,
        radii: vec![Rational::from(0); components],
        metadata,
    })
}
