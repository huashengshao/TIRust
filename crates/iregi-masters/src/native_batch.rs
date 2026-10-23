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
mod db0;
#[rustfmt::skip]
#[allow(clippy::all, unused_assignments, dead_code)]
#[path = "generated/c0.rs"]
mod c0;
#[rustfmt::skip]
#[allow(clippy::all, unused_assignments, dead_code)]
#[path = "generated/d0.rs"]
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
pub(crate) struct NativeBatchEvaluator<T: NativeFloat> {
    family: ScalarIntegral,
    context: Context<T>,
    workspace: BatchWorkspace<T>,
    policy: BatchExecutionPolicy,
}

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
