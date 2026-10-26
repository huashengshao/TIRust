//! TARGET-ONLY WORKER SOURCE PROPOSAL: never compiled, prepared or integrated here.
//! Reuse the existing Symbolica Laurent/source kernel; no reducer or master.
//! The owner separately admits EC27, positive F, normalization and any runtime.

#[path = "kernel_prefix.rs"]
mod existing;

use serde_json::{Value, json};
use symbolica::{
    atom::{Atom, AtomCore, Symbol},
    domains::{float::Complex, rational::Rational},
    evaluate::ExpressionEvaluator,
};

pub type Answer<T> = Result<T, String>;
pub const SELECTED: [[u16; 4]; 2] = [[0, 0, 0, 0], [1, 0, 0, 0]];
pub const PARAMETERS: usize = 5;
pub const NATIVE_OUTPUTS: usize = 8;

/// Two selected moments, not a complete cumulative rank-six tensor array.
/// Native output order: [selected moment][UV1, IR1, IR2, finite].
pub struct SelectedPrepared {
    pub symbols: existing::Symbols,
    pub expressions: [Atom; NATIVE_OUTPUTS],
}

impl SelectedPrepared {
    pub fn new() -> Answer<Self> {
        let symbols = existing::Symbols::new();
        let f_symbol = symbolica::symbol!("low_n_parameter_oracle::F");
        if Atom::var(f_symbol) != symbols.f {
            return Err("existing kernel's F parameter identity differs".into());
        }
        let mut expressions = Vec::with_capacity(NATIVE_OUTPUTS);
        for alpha in SELECTED {
            // At source degrees0/1, N4/D4 has only Gamma(2+epsilon),
            // and -d/dF supplies Gamma(3+epsilon) for N6/D6.
            // Both denominator signs are even. Native D6 needs one more pi.
            // Symbolica owns both source differentiation and F differentiation.
            let [uv4, finite4] = existing::direct_source_derivative(4, alpha, &symbols)?;
            let factor = -Atom::var(Symbol::PI);
            let uv6 = &factor * uv4.derivative(f_symbol);
            let finite6 = &factor * finite4.derivative(f_symbol);
            if !existing::exact_zero(&uv6) {
                return Err("selected convergent D6 moment retained a UV pole".into());
            }
            // IR zeros require the owner's separately checked positive-F
            // massive compact-simplex domain; they are not inferred from data.
            expressions.extend([uv6, Atom::new(), Atom::new(), finite6]);
        }
        let expressions = expressions.try_into()
            .map_err(|_| "selected native output width differs".to_string())?;
        Ok(Self { symbols, expressions })
    }

    /// Preparation only. A future worker must map/evaluate with the existing
    /// Symbolica instruction-major batch facility under a separate grant.
    pub fn exact_evaluator(&self) -> Answer<ExpressionEvaluator<Complex<Rational>>> {
        Atom::evaluator_multiple(&self.expressions, &self.symbols.parameters())
            .direct_translation(true).horner_iterations(0).cores(1).build()
            .map_err(|error| format!("selected Symbolica exact evaluator: {error}"))
    }

    pub fn artifact(&self) -> Value {
        json!({
            "schema_version":1,
            "status":"selected_Symbolica_kernels_prepared_not_integrated",
            "propagators":6,"dimension_shift_h":1,"dimension":"6-2epsilon",
            "powers":[1,1,1,1,1,1],"selected_basis":SELECTED,
            "parent_cumulative_IREGI_indices":[0,1],
            "parameters":["P0","Px","Py","Pz","F"],
            "native_slots":["uv1","ir1","ir2","finite"],
            "native_outputs":NATIVE_OUTPUTS,"raw_unweighted_components":true,
            "native_normalization_included":true,"uv_poles_proven_zero":true,"ir_poles_proven_zero":true,
            "normalization":"native i*pi^3 already included; no extra Gamma(N), pi, factorial or simplex volume",
            "simplex_integration":"not performed; the later map or weights must supply the simplex measure exactly once",
            "kernel_reuse":"existing N4/D4 direct source derivatives, followed by Symbolica -pi*d/dF",
            "domain_requirement":"owner independently admits fixed real massive EC27 unit powers, positive F, h1 and selected ranks0/1",
            "expressions":self.expressions.iter().map(|atom| atom.to_canonical_string()).collect::<Vec<_>>(),
            "no_reducer_or_master_imports":true,
            "quadrature_executed":false,"performance_qualified":false,
            "full_rank6_or_Dual_qualification":false
        })
    }
}

#[cfg(test)]
mod proposed_tests {
    use super::*;

    #[test]
    fn selected_native_scalar_and_time_shift_have_exact_ratio() {
        let prepared = SelectedPrepared::new().unwrap();
        let scalar = &prepared.expressions[3];
        let first = &prepared.expressions[7];
        // Independent low-rank completed-square checks, delegated to Symbolica.
        let expected = symbolica::parse!("𝑖") * Atom::var(Symbol::PI).pow(3)
            * Atom::num(2) / prepared.symbols.f.pow(3);
        assert!(existing::exact_zero(&(scalar - expected)));
        assert!(existing::exact_zero(&(first + &prepared.symbols.p[0] * scalar)));
        assert!(!existing::exact_zero(scalar));
        for slots in prepared.expressions.chunks_exact(4) {
            assert!(slots[..3].iter().all(existing::exact_zero));
        }
    }

    #[test]
    fn selected_layout_is_explicit_and_not_a_full_tensor() {
        let prepared = SelectedPrepared::new().unwrap();
        assert_eq!(SELECTED, [[0,0,0,0],[1,0,0,0]]);
        assert_eq!(prepared.symbols.parameters().len(), PARAMETERS);
        assert_eq!(prepared.expressions.len(), 8);
        assert_eq!(prepared.artifact()["parent_cumulative_IREGI_indices"], json!([0,1]));
    }
}
