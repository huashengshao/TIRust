//! Compiled scalar kernels emitted from the same exact six-output expressions
//! as the Symbolica evaluator. Runtime Dual shapes use the Symbolica backend.
//!
//! Regenerate the child modules with `--no-default-features --example
//! generate_batch` so generation also works before those modules exist.

use oneloop::{
    ScalarIntegral,
    rust_codegen::runtime::{
        BatchExecutionPolicy, BatchWorkspace, C, ConstantSpec, Context, NativeFloat, batch,
        primitives,
    },
};

#[rustfmt::skip]
#[allow(clippy::all, unused_assignments, dead_code)]
#[path = "generated/a0.rs"]
mod a0;
#[rustfmt::skip]
#[allow(clippy::all, unused_assignments, dead_code)]
#[path = "generated/b0.rs"]
mod b0;
#[rustfmt::skip]
#[allow(clippy::all, unused_assignments, dead_code)]
#[path = "generated/db0.rs"]
#[cfg(feature = "native-batch")]
mod db0;
#[rustfmt::skip]
#[allow(clippy::all, unused_assignments, dead_code)]
#[path = "generated/c0.rs"]
mod c0;
#[rustfmt::skip]
#[allow(clippy::all, unused_assignments, dead_code)]
#[path = "generated/d0.rs"]
#[cfg(feature = "native-batch")]
mod d0;
#[rustfmt::skip]
#[allow(clippy::all, unused_assignments, dead_code)]
#[path = "generated/constants.rs"]
mod constants;

/// Constants and retained scratch storage for one scalar master family.
///
/// Generated instructions iterate over active lanes. The default whole-batch
/// policy preserves that behavior across the complete input buffer. Optional
/// fixed blocks bound scratch storage without splitting individual rows.
#[derive(Clone)]
#[cfg(feature = "native-batch")]
pub(crate) struct NativeBatchEvaluator<T: NativeFloat> {
    family: ScalarIntegral,
    context: Context<T>,
    workspace: BatchWorkspace<T>,
    policy: BatchExecutionPolicy,
}

#[cfg(feature = "native-batch")]
impl<T: NativeFloat> NativeBatchEvaluator<T> {
    pub(crate) fn new(family: ScalarIntegral, prototype: &T) -> Result<Self, String> {
        Ok(Self {
            family,
            context: Context::new(constants::CONSTANTS, prototype.get_precision())?,
            workspace: BatchWorkspace::new(prototype.clone()),
            policy: BatchExecutionPolicy::Whole,
        })
    }

    pub(crate) fn set_policy(&mut self, policy: BatchExecutionPolicy) {
        self.policy = policy;
    }

    pub(crate) fn evaluate_batch(
        &mut self,
        input: &[C<T>],
        output: &mut [C<T>],
        rows: usize,
    ) -> Result<(), String> {
        let family = self.family;
        let context = &self.context;
        self.policy.evaluate(
            input,
            output,
            rows,
            (family.arity(), super::OUTPUTS),
            &mut self.workspace,
            |input, output, rows, workspace| match family {
                ScalarIntegral::A0 => a0::evaluate_batch(input, output, rows, context, workspace),
                ScalarIntegral::B0 => b0::evaluate_batch(input, output, rows, context, workspace),
                ScalarIntegral::DB0 => db0::evaluate_batch(input, output, rows, context, workspace),
                ScalarIntegral::C0 => c0::evaluate_batch(input, output, rows, context, workspace),
                ScalarIntegral::D0 => d0::evaluate_batch(input, output, rows, context, workspace),
            },
        )
    }
}

/// Concrete-f64 dispatch over the same generated A0/B0/C0 modules.
///
/// Keeping this separate from the generic all-family dispatcher avoids compiling
/// D0/dB0 for the selective feature. All validation and lane adaptation remains
/// in `MasterEvaluator`; this owns only numeric constants and native scratch.
#[cfg(feature = "native-f64-abc")]
#[derive(Clone)]
pub(crate) struct SelectedBatchEvaluator {
    family: ScalarIntegral,
    context: Context<f64>,
    workspace: BatchWorkspace<f64>,
    policy: BatchExecutionPolicy,
}

#[cfg(feature = "native-f64-abc")]
impl SelectedBatchEvaluator {
    pub(crate) fn new(family: ScalarIntegral) -> Result<Self, String> {
        if !matches!(family, ScalarIntegral::A0 | ScalarIntegral::B0 | ScalarIntegral::C0) {
            return Err("native-f64-abc supports ordinary f64 A0/B0/C0 only".into());
        }
        Ok(Self {
            family,
            context: Context::new(constants::CONSTANTS, 53)?,
            workspace: BatchWorkspace::new(0.),
            policy: BatchExecutionPolicy::Whole,
        })
    }

    pub(crate) fn set_policy(&mut self, policy: BatchExecutionPolicy) {
        self.policy = policy;
    }

    pub(crate) fn evaluate_batch<T: NativeFloat>(
        &mut self,
        input: &Vec<C<T>>,
        output: &mut Vec<C<T>>,
        rows: usize,
    ) -> Result<(), String> {
        let input = (input as &dyn std::any::Any)
            .downcast_ref::<Vec<C<f64>>>()
            .ok_or("native-f64-abc input type differs from f64")?;
        let output = (output as &mut dyn std::any::Any)
            .downcast_mut::<Vec<C<f64>>>()
            .ok_or("native-f64-abc output type differs from f64")?;
        let family = self.family;
        let context = &self.context;
        self.policy.evaluate(
            input,
            output,
            rows,
            (family.arity(), super::OUTPUTS),
            &mut self.workspace,
            |input, output, rows, workspace| match family {
                ScalarIntegral::A0 => a0::evaluate_batch(input, output, rows, context, workspace),
                ScalarIntegral::B0 => b0::evaluate_batch(input, output, rows, context, workspace),
                ScalarIntegral::C0 => c0::evaluate_batch(input, output, rows, context, workspace),
                _ => unreachable!("selected family validated at construction"),
            },
        )
    }
}
