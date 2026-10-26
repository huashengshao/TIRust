//! Target-only exact Symbolica evaluator reuse; no reducer/master imports.
use crate::Answer;
use serde_json::{Value,json};
use std::collections::BTreeSet;
use symbolica::{atom::{Atom,AtomCore,Symbol},domains::{float::{Complex,Float,FloatLike,Real,SingleFloat},rational::Rational},
    evaluate::{BatchExpressionEvaluator,ExpressionEvaluator,ExportedInstructions,Instruction,Slot}};
pub const MAX_SLOT_VALUES:usize=4_000_000;
pub const PARAMETERS:usize=5;
#[derive(Debug)]
pub struct Inventory {
    pub inputs: usize,
    pub constants: usize,
    pub temporaries: usize,
    pub outputs: usize,
    pub slots: usize,
    pub instructions: usize,
    pub dynamic_logarithms: usize,
    pub constant_functions: Vec<Value>,
}

/// Audit the exported layout before mapping registered constants. In particular,
/// permitting pi/Euler constants must not silently permit arbitrary callbacks.
/// This is instruction/layout validation, not symbolic algebra.
fn audit(exported: &ExportedInstructions<Complex<Rational>>, expected_outputs: usize) -> Answer<Inventory> {
    if exported.input_count != PARAMETERS || exported.output_count != expected_outputs
        || !exported.sub_evaluators.is_empty()
    {
        return Err("unexpected evaluator dimensions or shared sub-evaluators".into());
    }
    let mut definitions = BTreeSet::new();
    let mut constant_functions = Vec::new();
    for definition in &exported.constant_functions {
        if definition.index >= exported.constants.len() || !definitions.insert(definition.index)
            || !definition.tags.is_empty()
        {
            return Err("invalid or duplicate registered constant slot/tags".into());
        }
        let known_scalar = (definition.symbol == Symbol::PI
            || definition.symbol == symbolica::transcendental::euler_gamma())
            && definition.fixed_args.is_empty();
        let positive_rational_log = definition.symbol == Symbol::LOG
            && definition.fixed_args.len() == 1
            && definition.fixed_args[0].im == Rational::from(0)
            && definition.fixed_args[0].re > Rational::from(0);
        if !known_scalar && !positive_rational_log {
            return Err(format!("unapproved registered constant {}", definition.symbol));
        }
        constant_functions.push(json!({
            "index":definition.index,"symbol":definition.symbol.to_string(),
            "tags":definition.tags,
            "fixed_args":definition.fixed_args.iter().map(|v|
                [v.re.to_string(),v.im.to_string()]).collect::<Vec<_>>()
        }));
    }
    let slots = exported.input_count.checked_add(exported.constants.len())
        .and_then(|n| n.checked_add(exported.temporary_count))
        .and_then(|n| n.checked_add(exported.output_count))
        .ok_or("evaluator slot count overflow")?;
    if slots > MAX_SLOT_VALUES {
        return Err("evaluator exceeds fixed numeric-slot allowance".into());
    }
    // A bit mask records numerical input dependence; None is uninitialized.
    let mut temps = vec![None; exported.temporary_count];
    let mut outputs = vec![None; exported.output_count];
    let read = |slot: Slot, temps: &[Option<u8>], outputs: &[Option<u8>]| -> Answer<u8> {
        match slot {
            Slot::Param(i) if i < exported.input_count => Ok(1 << i),
            Slot::Const(i) if i < exported.constants.len() => Ok(0),
            Slot::Temp(i) => temps.get(i).copied().flatten()
                .ok_or_else(|| "uninitialized or out-of-range temporary read".into()),
            Slot::Out(i) => outputs.get(i).copied().flatten()
                .ok_or_else(|| "uninitialized or out-of-range output read".into()),
            _ => Err("out-of-range parameter or constant read".into()),
        }
    };
    let mut dynamic_logarithms = 0;
    for instruction in &exported.instructions {
        let (destination, dependence) = match instruction {
            Instruction::Add(out, args, real) | Instruction::Mul(out, args, real) => {
                if args.is_empty() || *real > args.len() {
                    return Err("invalid arithmetic instruction arguments".into());
                }
                let mut dependence = 0;
                for &slot in args { dependence |= read(slot, &temps, &outputs)?; }
                (*out, dependence)
            }
            Instruction::Pow(out, base, _, _) | Instruction::Assign(out, base) =>
                (*out, read(*base, &temps, &outputs)?),
            Instruction::Fun(out, call, _) => {
                let (symbol, tags, args) = call.as_ref();
                if *symbol != Symbol::LOG || !tags.is_empty() || args.len() != 1 {
                    return Err("only the built-in single-argument logarithm is allowed".into());
                }
                let dependence = read(args[0], &temps, &outputs)?;
                if dependence != 0 {
                    // Dynamic logarithms are the canonical log(F) only. Constant
                    // log(pi), log(9), etc. may pass through constant arithmetic.
                    if args[0] != Slot::Param(4) {
                        return Err("dynamic logarithm argument is not the F parameter".into());
                    }
                    dynamic_logarithms += 1;
                }
                (*out, dependence)
            }
            _ => return Err("unexpected noninteger power or control-flow instruction".into()),
        };
        match destination {
            Slot::Temp(i) => *temps.get_mut(i).ok_or("out-of-range temporary write")? = Some(dependence),
            Slot::Out(i) => *outputs.get_mut(i).ok_or("out-of-range output write")? = Some(dependence),
            _ => return Err("instruction attempts to overwrite parameter or constant".into()),
        }
    }
    if outputs.iter().any(Option::is_none) {
        return Err("evaluator leaves an output uninitialized".into());
    }
    Ok(Inventory { inputs:exported.input_count,constants:exported.constants.len(),
        temporaries:exported.temporary_count,outputs:exported.output_count,
        slots, instructions:exported.instructions.len(), dynamic_logarithms,
        constant_functions })
}

pub struct Program { pub exact:ExpressionEvaluator<Complex<Rational>>,pub inventory:Inventory,
    expressions:Vec<String>,parameters:Vec<String> }
impl Program {
    pub fn new(expressions:&[Atom],parameters:&[Atom])->Answer<Self> {
        if expressions.is_empty() || parameters.len()!=5 {return Err("invalid exact program dimensions".into());}
        let exact=Atom::evaluator_multiple(expressions,parameters).direct_translation(true)
            .horner_iterations(0).cores(1).build().map_err(|e|e.to_string())?;
        let inventory=audit(&exact.export_instructions(),expressions.len())?;
        Ok(Self{exact,inventory,expressions:expressions.iter().map(|a|a.to_canonical_string()).collect(),
            parameters:parameters.iter().map(|a|a.to_canonical_string()).collect()})
    }
    pub fn instantiate(&self)->Answer<BatchExpressionEvaluator<Complex<Float>>> {
        let prototype=Float::new(128);
        let mapped=self.exact.clone().map_coeff_with_prec(&|v|Complex::new(prototype.from_rational(&v.re),prototype.from_rational(&v.im)),128);
        if mapped.export_instructions().constants.iter().any(|v|!v.is_finite()) {return Err("nonfinite Float128 constant".into());}
        Ok(mapped.into_batch_evaluator())
    }
    pub fn artifact(&self)->Value {json!({"instructions":self.inventory.instructions,"numeric_slots":self.inventory.slots,
        "inputs":self.inventory.inputs,"constants":self.inventory.constants,
        "temporaries":self.inventory.temporaries,"outputs":self.inventory.outputs,
        "parameters":self.parameters,"expressions":self.expressions,
        "dynamic_logarithms":self.inventory.dynamic_logarithms,"constant_functions":self.inventory.constant_functions})}
}
