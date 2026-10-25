//! Private validation transport; no reducer, scalar master, JIT or foreign ABI.
//!
//! The wire stores little-endian `(real, imaginary)` f64 pairs. Rust workspaces
//! retain their numeric domain: Float coefficients and constants are mapped at
//! the prototype precision, without an intermediate f64 conversion.
use super::{Answer, Prepared};
use serde_json::{Value, json};
use std::{collections::BTreeSet, io::{self, Read, Write}};
use symbolica::{
    atom::{AtomCore, Symbol},
    domains::{float::{Complex, Float, FloatLike, Real, SingleFloat}, rational::Rational},
    evaluate::{BatchExpressionEvaluator, ExportedInstructions, Instruction, Slot},
    evaluate::EvaluationDomain,
};

pub const MOMENTS: usize = 330;
pub const PARAMETERS: usize = 5;
pub const NATIVE_SLOTS: usize = 4;
pub const OUTPUTS: usize = MOMENTS * NATIVE_SLOTS;
pub const MAX_ROWS: usize = 256;
/// Complex numeric slots, including a conservative allowance for copied output.
/// This bounds value counts, not the heap size of arbitrary-precision values.
pub const MAX_SLOT_VALUES: usize = 4_000_000;
pub const HEADER_BYTES: usize = 9;
pub const COMPLEX_BYTES: usize = 16;
pub const MAX_REQUEST: usize = HEADER_BYTES + MAX_ROWS * PARAMETERS * COMPLEX_BYTES;
pub const MAX_RESPONSE: usize = HEADER_BYTES + MAX_ROWS * OUTPUTS * COMPLEX_BYTES;
pub const OP_EVALUATE: u8 = 1;
pub const OP_QUIT: u8 = 2;
pub const RESPONSE_VALUES: u8 = 0;
pub const RESPONSE_ERROR: u8 = 1;
pub const RESPONSE_READY: u8 = 2;
pub const RESPONSE_QUIT: u8 = 3;
pub const PROTOCOL: &str = "parameter-oracle-le-complex-f64-v2";

#[derive(Debug)]
struct Inventory {
    slots: usize,
    instructions: usize,
    dynamic_logarithms: usize,
    constant_functions: Vec<Value>,
}

/// Audit the exported layout before mapping registered constants. In particular,
/// permitting pi/Euler constants must not silently permit arbitrary callbacks.
/// This is instruction/layout validation, not symbolic algebra.
fn audit(exported: &ExportedInstructions<Complex<Rational>>) -> Answer<Inventory> {
    if exported.input_count != PARAMETERS || exported.output_count != OUTPUTS
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
    Ok(Inventory { slots, instructions:exported.instructions.len(), dynamic_logarithms,
        constant_functions })
}

pub struct Workspace<T: Real + SingleFloat>
where Complex<T>: EvaluationDomain {
    evaluator: BatchExpressionEvaluator<Complex<T>>,
    prototype: T,
    topology: usize,
    pub instructions: usize,
    pub numeric_slots_upper_bound: usize,
    pub max_rows: usize,
    pub batches: u64,
    pub rows: u64,
    /// Accounting for the exported instruction schedule, including synthetic
    /// output copies. This is not instrumentation of Symbolica's interpreter.
    pub scheduled_instruction_lane_visits: u64,
    dynamic_logarithms: usize,
    constant_functions: Vec<Value>,
    resolved_constants: Vec<[String; 2]>,
}
pub type FloatWorkspace = Workspace<Float>;

impl Workspace<f64> {
    pub fn new(prepared: &Prepared) -> Answer<Self> { Self::with_precision(prepared, 0.0) }
}

impl<T: Real + SingleFloat> Workspace<T>
where Complex<T>: EvaluationDomain {
    pub fn with_precision(prepared: &Prepared, prototype: T) -> Answer<Self> {
        if !(2..=4).contains(&prepared.n) || prepared.symbols.parameters().len() != PARAMETERS
            || prepared.expressions.len() != OUTPUTS
        {
            return Err("complex oracle admits N2/N3/N4 and exactly 1320 native outputs".into());
        }
        if prepared.expressions.chunks_exact(NATIVE_SLOTS)
            .any(|moment| !moment[1].is_zero() || !moment[2].is_zero())
        {
            return Err("restricted positive-real-part oracle requires exact IR zero atoms".into());
        }
        if !prototype.is_finite() || prototype.get_precision() == 0 {
            return Err("numeric prototype must be finite with positive precision".into());
        }
        let exact = prepared.exact_evaluator()?;
        let inventory = audit(&exact.export_instructions())?;
        // Symbolica resolves registered pi/Euler/fixed log constants here using
        // this precision; rational real and imaginary parts are mapped directly.
        let mapped = exact.map_coeff_with_prec(&|value| Complex::new(
            prototype.from_rational(&value.re), prototype.from_rational(&value.im)),
            prototype.get_precision());
        let mapped_constants = mapped.export_instructions().constants;
        if mapped_constants.iter().any(|value| !value.is_finite()) {
            return Err("nonfinite mapped evaluator constant".into());
        }
        let resolved_constants = mapped_constants.iter()
            .map(|value| [value.re.to_string(),value.im.to_string()]).collect();
        let max_rows = MAX_ROWS.min(MAX_SLOT_VALUES / inventory.slots.max(1));
        if max_rows == 0 { return Err("no row fits numeric-slot allowance".into()); }
        Ok(Self { evaluator:mapped.into_batch_evaluator(), prototype, topology:prepared.n,
            instructions:inventory.instructions,numeric_slots_upper_bound:inventory.slots,
            max_rows,batches:0,rows:0,scheduled_instruction_lane_visits:0,
            dynamic_logarithms:inventory.dynamic_logarithms,
            constant_functions:inventory.constant_functions,resolved_constants })
    }

    pub fn evaluate(&mut self, input: &[Complex<T>], rows: usize) -> Answer<Vec<Complex<T>>> {
        let expected = rows.checked_mul(PARAMETERS).ok_or("input size overflow")?;
        if rows > self.max_rows || input.len() != expected {
            return Err("point batch exceeds declared dimensions or numeric-slot cap".into());
        }
        let zero = self.prototype.zero();
        let one = self.prototype.one();
        for row in input.chunks_exact(PARAMETERS) {
            if row.iter().any(|value| !value.is_finite())
                || row[..4].iter().any(|value| !value.im.is_zero())
                || !matches!(row[4].re.real_cmp(&one), Some(std::cmp::Ordering::Equal | std::cmp::Ordering::Greater))
                || !matches!(row[4].im.real_cmp(&zero), Some(std::cmp::Ordering::Equal | std::cmp::Ordering::Less))
            {
                return Err("points require finite real P, Re(F)>=1 and Im(F)<=0".into());
            }
        }
        // Check the full success accounting before executing or mutating counters.
        let rows_u64 = u64::try_from(rows).map_err(|_| "row counter conversion overflow")?;
        let next_batches = self.batches.checked_add(1).ok_or("batch counter overflow")?;
        let next_rows = self.rows.checked_add(rows_u64).ok_or("row counter overflow")?;
        let visits = u64::try_from(self.instructions).map_err(|_| "instruction counter overflow")?
            .checked_mul(rows_u64).ok_or("instruction lane count overflow")?;
        let next_visits = self.scheduled_instruction_lane_visits.checked_add(visits)
            .ok_or("instruction lane counter overflow")?;
        let output_size = rows.checked_mul(OUTPUTS).ok_or("output size overflow")?;
        let mut output = vec![Complex::new(zero.clone(),zero); output_size];
        self.evaluator.evaluate_batch(input,&mut output,rows)
            .map_err(|error| format!("Symbolica complex batch evaluation: {error}"))?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err("nonfinite native Laurent-kernel output; entire batch rejected".into());
        }
        if output.chunks_exact(NATIVE_SLOTS)
            .any(|moment| !moment[1].is_zero() || !moment[2].is_zero())
        {
            return Err("numeric evaluator did not preserve exact IR zero slots".into());
        }
        self.batches = next_batches;
        self.rows = next_rows;
        self.scheduled_instruction_lane_visits = next_visits;
        Ok(output)
    }

    pub fn ready(&self, n: usize) -> Value {
        json!({"protocol":PROTOCOL,"topology":self.topology,"requested_topology":n,
            "parameters":PARAMETERS,"moments":MOMENTS,"native_slots":NATIVE_SLOTS,
            "outputs":OUTPUTS,"complex_layout":"interleaved_real_imaginary",
            "output_order":"moment_major_native_uv1_ir1_ir2_finite",
            "native_normalization_included":true,"ir_poles_proven_zero":true,
            "max_rows":self.max_rows,"max_request_bytes":MAX_REQUEST,
            "max_response_bytes":MAX_RESPONSE,"frame_prefix_bytes":8,
            "byte_bounds_exclude_length_prefix":true,
            "instruction_major_symbolica_batch":true,"no_jit":true,
            "instructions":self.instructions,"dynamic_logarithms":self.dynamic_logarithms,
            "numeric_slots_upper_bound":self.numeric_slots_upper_bound,
            "max_slot_values":MAX_SLOT_VALUES,"slot_unit":"complex numeric values; not heap bytes",
            "working_precision_bits":self.prototype.get_precision(),
            "constant_mapping":"Symbolica map_coeff_with_prec, rational parts mapped directly",
            "constant_functions":self.constant_functions,"resolved_constants":self.resolved_constants,
            "domain":"finite real P; Re(F)>=1; Im(F)<=0",
            "successful_batches":self.batches,"successful_rows":self.rows,
            "scheduled_instruction_lane_visits":self.scheduled_instruction_lane_visits,
            "counter_scope":"exported schedule accounting, not interpreter instrumentation"})
    }
}

/// Length-prefixed payload. EOF is accepted only between complete frames.
pub fn read_frame<R: Read>(input: &mut R, cap: usize) -> Answer<Option<Vec<u8>>> {
    let mut prefix = [0_u8;8];
    let mut have = 0;
    while have < prefix.len() {
        match input.read(&mut prefix[have..]) {
            Ok(0) if have == 0 => return Ok(None),
            Ok(0) => return Err("truncated frame length".into()),
            Ok(n) => have += n,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(format!("read frame length: {error}")),
        }
    }
    let size = usize::try_from(u64::from_le_bytes(prefix)).map_err(|_| "frame size overflow")?;
    if !(HEADER_BYTES..=cap).contains(&size) {
        return Err("frame length outside fixed bounds".into());
    }
    let mut payload = vec![0;size];
    input.read_exact(&mut payload).map_err(|error| format!("truncated frame payload: {error}"))?;
    Ok(Some(payload))
}

pub fn write_frame<W: Write>(output: &mut W, payload: &[u8]) -> Answer<()> {
    if !(HEADER_BYTES..=MAX_RESPONSE).contains(&payload.len()) {
        return Err("outgoing frame length outside fixed bounds".into());
    }
    let length = u64::try_from(payload.len()).map_err(|_| "outgoing frame length overflow")?;
    output.write_all(&length.to_le_bytes()).map_err(|error| error.to_string())?;
    output.write_all(payload).map_err(|error| error.to_string())?;
    output.flush().map_err(|error| error.to_string())
}

pub fn message(code: u8, rows: usize, columns: usize, body: &[u8]) -> Answer<Vec<u8>> {
    let row_header = u32::try_from(rows).map_err(|_| "row header overflow")?;
    let column_header = u32::try_from(columns).map_err(|_| "column header overflow")?;
    let size = HEADER_BYTES.checked_add(body.len()).ok_or("message size overflow")?;
    if size > MAX_RESPONSE { return Err("message exceeds fixed response bound".into()); }
    let mut payload = Vec::with_capacity(size);
    payload.push(code);
    payload.extend_from_slice(&row_header.to_le_bytes());
    payload.extend_from_slice(&column_header.to_le_bytes());
    payload.extend_from_slice(body);
    Ok(payload)
}

fn complex_bytes(values: &[Complex<f64>]) -> Vec<u8> {
    values.iter().flat_map(|value| value.re.to_le_bytes().into_iter()
        .chain(value.im.to_le_bytes())).collect()
}

fn decode_complex(bytes: &[u8]) -> Answer<Vec<Complex<f64>>> {
    if bytes.len() % COMPLEX_BYTES != 0 { return Err("partial complex wire value".into()); }
    bytes.chunks_exact(COMPLEX_BYTES).map(|pair| {
        let re = f64::from_le_bytes(pair[..8].try_into().map_err(|_| "real wire length")?);
        let im = f64::from_le_bytes(pair[8..].try_into().map_err(|_| "imaginary wire length")?);
        if !re.is_finite() || !im.is_finite() { return Err("nonfinite complex wire value".into()); }
        Ok(Complex::new(re,im))
    }).collect()
}

/// Request helper checks shape; domain-invalid finite values are deliberately
/// left for the worker, so error transport can be tested independently.
pub fn request_values(rows: usize, values: &[Complex<f64>]) -> Answer<Vec<u8>> {
    let expected = rows.checked_mul(PARAMETERS).ok_or("input size overflow")?;
    if rows > MAX_ROWS || values.len() != expected { return Err("invalid request helper dimensions".into()); }
    message(OP_EVALUATE,rows,PARAMETERS,&complex_bytes(values))
}

/// Client-side complete-response validation. An empty body is valid only for an
/// explicitly shaped zero-row VALUES response; EOF/null/error is never a result.
pub fn decode_values_response(payload: &[u8], expected_rows: usize) -> Answer<Vec<Complex<f64>>> {
    let size = expected_rows.checked_mul(OUTPUTS).and_then(|n| n.checked_mul(COMPLEX_BYTES))
        .and_then(|n| n.checked_add(HEADER_BYTES)).ok_or("response size overflow")?;
    if expected_rows > MAX_ROWS || payload.len() != size || payload.len() < HEADER_BYTES {
        return Err("missing, partial or oversized values response".into());
    }
    let rows = u32::from_le_bytes(payload[1..5].try_into().map_err(|_| "response row header")?) as usize;
    let columns = u32::from_le_bytes(payload[5..9].try_into().map_err(|_| "response column header")?) as usize;
    if payload[0] != RESPONSE_VALUES || rows != expected_rows || columns != OUTPUTS {
        return Err("response opcode or shape mismatch".into());
    }
    decode_complex(&payload[HEADER_BYTES..])
}

pub fn serve<R: Read, W: Write>(work: &mut Workspace<f64>, n: usize,
    input: &mut R, output: &mut W) -> Answer<()> {
    if n != work.topology { return Err("worker topology argument disagrees with prepared kernel".into()); }
    let ready = serde_json::to_vec(&work.ready(n)).map_err(|error| error.to_string())?;
    write_frame(output,&message(RESPONSE_READY,0,0,&ready)?)?;
    loop {
        let request = match read_frame(input,MAX_REQUEST) {
            Ok(Some(request)) => request,
            Ok(None) => return Ok(()),
            Err(error) => {
                write_frame(output,&message(RESPONSE_ERROR,0,0,error.as_bytes())?)?;
                return Err(error);
            }
        };
        let rows = u32::from_le_bytes(request[1..5].try_into().map_err(|_| "request row header")?) as usize;
        let columns = u32::from_le_bytes(request[5..9].try_into().map_err(|_| "request column header")?) as usize;
        let result = (|| -> Answer<Option<Vec<Complex<f64>>>> {
            if request[0] == OP_QUIT && rows == 0 && columns == 0 && request.len() == HEADER_BYTES {
                return Ok(None);
            }
            let expected = rows.checked_mul(PARAMETERS).and_then(|n| n.checked_mul(COMPLEX_BYTES))
                .and_then(|n| n.checked_add(HEADER_BYTES)).ok_or("request size overflow")?;
            if request[0] != OP_EVALUATE || columns != PARAMETERS || rows > work.max_rows
                || request.len() != expected {
                return Err("invalid request opcode, shape or payload size".into());
            }
            work.evaluate(&decode_complex(&request[HEADER_BYTES..])?,rows).map(Some)
        })();
        match result {
            Ok(None) => {
                write_frame(output,&message(RESPONSE_QUIT,0,0,&[])?)?;
                return Ok(());
            }
            Ok(Some(values)) => write_frame(output,
                &message(RESPONSE_VALUES,rows,OUTPUTS,&complex_bytes(&values))?)?,
            Err(error) => {
                write_frame(output,&message(RESPONSE_ERROR,0,0,error.as_bytes())?)?;
                return Err(error);
            }
        }
    }
}

#[cfg(test)]
#[path="transport_tests.rs"]
mod tests;
