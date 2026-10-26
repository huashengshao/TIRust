//! Pre-recurrence query inventory for a conservative highpoint Legacy no-op gate.
//! Target-only, uncompiled: this does not construct a reduction graph or execute
//! any evaluator. LegacySnapping owns every policy predicate and query result.
//! A complete finite superset is intentional: a changed unused query may be
//! rejected, while an accepted row proves equivalence for all possible calls.
use crate::snapping::{LegacySnapping, SnapCompatibilityQuery, SnapDiagnostic, exact_threshold};
use crate::{InputLayout, InputPolicy, InvariantInput, ReductionError, ReductionSettings};
use symbolica::atom::Atom;

#[derive(Clone, Debug)]
pub(crate) struct LegacyNoOpQueries {
    pub(crate) parameters: Vec<Atom>,
    pub(crate) layout: InputLayout,
    pub(crate) aliases: Vec<(Atom, Atom)>,
    pub(crate) queries: Vec<SnapCompatibilityQuery>,
    pub(crate) diagnostics: Vec<SnapDiagnostic>,
    pub(crate) threshold: Atom,
    pub(crate) preserve_settings: ReductionSettings,
}

impl LegacyNoOpQueries {
    pub(crate) fn prepare(settings: &ReductionSettings) -> Result<Self, ReductionError> {
        let n = settings.propagators;
        if !(5..=6).contains(&n)
            || settings.rank > 6
            || settings.invariants != InvariantInput::FromShifts
            || settings.powers.len() != n
            || settings.powers.contains(&0)
        {
            return Err(ReductionError::Unsupported(
                "legacy no-op frontstage requires N5/N6 rank<=6 Cartesian positive-power input"
                    .into(),
            ));
        }
        let InputPolicy::LegacyOnShell { threshold } = &settings.input_policy else {
            return Err(ReductionError::InvalidInput(
                "legacy no-op frontstage requires the requested LegacyOnShell policy".into(),
            ));
        };
        let threshold_atom = exact_threshold(threshold)?;
        let shifts: Vec<[Atom; 4]> = (0..n)
            .map(|i| {
                std::array::from_fn(|mu| Atom::var(symbolica::symbol!(format!("iregi::p{i}_{mu}"))))
            })
            .collect();
        let masses: Vec<_> = (0..n)
            .map(|i| Atom::var(symbolica::symbol!(format!("iregi::m2_{i}"))))
            .collect();
        let mut parameters: Vec<_> = shifts.iter().flatten().cloned().collect();
        parameters.extend(masses.iter().cloned());
        parameters.push(Atom::var(symbolica::symbol!("iregi::mu")));
        let layout = InputLayout {
            shifts: 0..4 * n,
            scalar_products: None,
            masses: 4 * n..5 * n,
            scale: 5 * n,
        };
        let mut policy = LegacySnapping::new(&masses, threshold)?;
        // Includes a possible nonzero zero-image at threshold0 and signed
        // reference masses; do not assume that snapping zero is an identity.
        policy.snap_scalar_product(Atom::new(), "conservative zero-diagonal query");
        for i in 0..n {
            for j in i + 1..n {
                policy.snap_scalar_product(
                    crate::plan::squared_difference(&shifts[i], &shifts[j]),
                    format!("conservative distance ({i},{j})"),
                );
            }
        }
        // pv::geometry uses only these raw dot products before its factor2,
        // with an original denominator as origin, including pinched/reversed
        // canonical requests. Reuse its EXACT expression builder, not a
        // polarization identity with a different floating evaluation graph.
        for origin in 0..n {
            let relative: Vec<[Atom; 4]> = shifts
                .iter()
                .map(|p| std::array::from_fn(|mu| &p[mu] - &shifts[origin][mu]))
                .collect();
            for i in 0..n {
                if i == origin {
                    continue;
                }
                for j in i..n {
                    if j == origin {
                        continue;
                    }
                    policy.snap_scalar_product(
                        crate::pv::dot(&relative[i], &relative[j]),
                        format!("conservative Gram ({i}-{origin},{j}-{origin})"),
                    );
                }
            }
        }
        let mut preserve_settings = settings.clone();
        preserve_settings.input_policy = InputPolicy::Preserve;
        Ok(Self {
            parameters,
            layout,
            aliases: policy.aliases().to_vec(),
            queries: policy.compatibility_queries().to_vec(),
            diagnostics: policy.diagnostics().to_vec(),
            threshold: threshold_atom,
            preserve_settings,
        })
    }
}
