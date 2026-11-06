//! Opt-in qualification of Symbolica's existing ordinary-root stack allocator
//! (oneloop's `OneLoopExpressions::evaluator_with_stack_reuse`; moved here from
//! the former vendored oneloopmaster copy, since its inputs are TIRust's).
//!
//! Run with an exclusive Symbolica lease and an explicit artifact directory:
//! stack_reuse_probe target/stack-reuse [A0|B0|dB0|C0|D0] [--emit]
//! No production generated file is replaced. No timing is claimed by this probe.

use oneloop::{
    NativeFloat, OneLoopExpressions, ScalarIntegral,
    rust_codegen::{EmitOptions, RustEmitter},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};
use symbolica::{
    domains::{
        float::{Complex, DoubleFloat, Float},
        rational::Rational,
    },
    evaluate::{EvaluationDomain, ExportedInstructions, ExpressionEvaluator, Instruction, Slot},
};

type Exact = ExpressionEvaluator<Complex<Rational>>;
type Graph = ExportedInstructions<Complex<Rational>>;

fn write(path: impl AsRef<Path>, bytes: impl AsRef<[u8]>) -> Result<(), String> {
    fs::write(path, bytes).map_err(|error| error.to_string())
}

// This is instruction provenance accounting, not a graph transformation. Read
// every source before recording a destination: allocated slots can be reused
// by the same instruction, including slots named Out by the exporter.
fn provenance(graph: &Graph) -> Result<String, String> {
    let mut bindings: BTreeMap<Slot, String> = (0..graph.input_count)
        .map(|i| (Slot::Param(i), format!("p{i}")))
        .chain((0..graph.constants.len()).map(|i| (Slot::Const(i), format!("c{i}"))))
        .collect();
    let mut output = format!(
        "inputs={} outputs={}\nconstants={:?}\nregistered={:?}\n",
        graph.input_count, graph.output_count, graph.constants, graph.constant_functions,
    );
    for (pc, instruction) in graph.instructions.iter().enumerate() {
        let source = |slot: &Slot| {
            bindings
                .get(slot)
                .cloned()
                .ok_or_else(|| format!("undefined source {slot} at instruction {pc}"))
        };
        let sources = |slots: &[Slot]| slots.iter().map(source).collect::<Result<Vec<_>, _>>();
        let (destination, description) = match instruction {
            Instruction::Add(dst, args, real) => {
                (Some(*dst), format!("add {:?} real={real}", sources(args)?))
            }
            Instruction::Mul(dst, args, real) => {
                (Some(*dst), format!("mul {:?} real={real}", sources(args)?))
            }
            Instruction::Pow(dst, base, power, real) => (
                Some(*dst),
                format!("pow {} {power} real={real}", source(base)?),
            ),
            Instruction::Powf(dst, base, power, real) => (
                Some(*dst),
                format!("powf {} {} real={real}", source(base)?, source(power)?),
            ),
            Instruction::Fun(dst, call, real) => (
                Some(*dst),
                format!(
                    "call {} {:?} {:?} real={real}",
                    call.0.get_name(),
                    call.1,
                    sources(&call.2)?,
                ),
            ),
            Instruction::Assign(dst, src) => (Some(*dst), format!("assign {}", source(src)?)),
            Instruction::Join(dst, condition, yes, no) => (
                Some(*dst),
                format!(
                    "join {} {} {}",
                    source(condition)?,
                    source(yes)?,
                    source(no)?
                ),
            ),
            Instruction::IfElse(condition, label) => {
                (None, format!("if {} goto {label}", source(condition)?))
            }
            Instruction::Goto(label) => (None, format!("goto {label}")),
            Instruction::Label(label) => (None, format!("label {label}")),
        };
        writeln!(output, "v{pc}: {description}").unwrap();
        if let Some(destination) = destination {
            bindings.insert(destination, format!("v{pc}"));
        }
    }
    for i in 0..graph.output_count {
        writeln!(output, "output {i} = {:?}", bindings.get(&Slot::Out(i))).unwrap();
    }
    Ok(output)
}

fn helpers(
    graph: &Graph,
    found: &mut BTreeMap<String, (usize, usize, String)>,
) -> Result<(), String> {
    for helper in &graph.sub_evaluators {
        let name = format!("{} {:?}", helper.symbol.get_name(), helper.tags);
        if found.contains_key(&name) {
            continue;
        }
        found.insert(
            name,
            (
                helper.instructions.temporary_count,
                helper.instructions.instructions.len(),
                provenance(&helper.instructions)?,
            ),
        );
        helpers(&helper.instructions, found)?;
    }
    Ok(())
}

fn point(values: &[f64], masses: std::ops::Range<usize>, complex: bool) -> Vec<Complex<f64>> {
    values
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            Complex::new(
                value,
                if complex && masses.contains(&i) {
                    -0.1
                } else {
                    0.
                },
            )
        })
        .collect()
}

fn cases(family: ScalarIntegral) -> Vec<Vec<Complex<f64>>> {
    match family {
        ScalarIntegral::A0 => vec![point(&[4., 1.], 0..1, false), point(&[4., 1.], 0..1, true)],
        ScalarIntegral::B0 | ScalarIntegral::DB0 => vec![
            point(&[-1., 4., 9., 1.], 1..3, false),
            point(&[30., 2., 3., 1.], 1..3, false),
            point(&[1., 4., 9., 1.], 1..3, true),
            point(&[-1., 0., 0., 1.], 1..3, false),
        ],
        ScalarIntegral::C0 => vec![
            point(&[1., 2., 3., 5., 7., 11., 1.], 3..6, false),
            point(&[-1., -2., 30., 2., 3., 5., 1.], 3..6, false),
            point(&[1., 2., 3., 0.5, 0.7, 1.4, 1.], 3..6, true),
            point(&[-1., -2., 30., 2., 3., 5., 1.], 3..6, true),
        ],
        ScalarIntegral::D0 => vec![
            point(
                &[1., 1., 1., 1., 8., -2., 20., 30., 40., 50., 1.],
                6..10,
                false,
            ),
            point(
                &[1., 1., 1., 1., 40., -8., 2., 3., 4., 5., 1.],
                6..10,
                false,
            ),
            point(
                &[-1., -2., -3., -4., 12., -5., 2., 3., 4., 5., 1.],
                6..10,
                true,
            ),
            point(
                &[-1., -2., -3., -4., 12., -5., 2., 2., 2., 2., 1.],
                6..10,
                true,
            ),
        ],
    }
}

fn same_value<T: NativeFloat>(a: &T, b: &T) -> bool {
    a.is_finite()
        && b.is_finite()
        && a.exact_rational() == b.exact_rational()
        && a.get_precision() == b.get_precision()
        && (!a.is_zero() || a.negative_sign() == b.negative_sign())
}

fn verify<T: NativeFloat>(
    label: &str,
    baseline: &Exact,
    candidate: &Exact,
    points: &[Vec<Complex<f64>>],
    prototype: T,
    row_counts: &[usize],
    log: &mut String,
) -> Result<(), String>
where
    Complex<T>: EvaluationDomain,
{
    let convert = |value: &Complex<Rational>| {
        Complex::new(
            prototype.from_rational(&value.re),
            prototype.from_rational(&value.im),
        )
    };
    let mut a = baseline
        .clone()
        .map_coeff_with_prec(&convert, prototype.get_precision())
        .into_batch_evaluator();
    let mut b = candidate
        .clone()
        .map_coeff_with_prec(&convert, prototype.get_precision())
        .into_batch_evaluator();
    for &rows in row_counts {
        for ordering in ["homogeneous", "interleaved", "grouped", "reversed"] {
            let input: Vec<_> = (0..rows)
                .flat_map(|lane| {
                    let index = match ordering {
                        "homogeneous" => 0,
                        "interleaved" => lane % points.len(),
                        "grouped" => (lane * points.len() / rows).min(points.len() - 1),
                        _ => (rows - 1 - lane) % points.len(),
                    };
                    points[index].iter().map(|value| {
                        Complex::new(
                            prototype.from_rational(&value.re.exact_rational().unwrap()),
                            prototype.from_rational(&value.im.exact_rational().unwrap()),
                        )
                    })
                })
                .collect();
            let zero = Complex::new(prototype.zero(), prototype.zero());
            let mut expected = vec![zero.clone(); rows * 3];
            let mut actual = vec![zero; rows * 3];
            a.evaluate_batch(&input, &mut expected, rows)
                .map_err(|error| error.to_string())?;
            b.evaluate_batch(&input, &mut actual, rows)
                .map_err(|error| error.to_string())?;
            for (slot, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
                if !same_value(&expected.re, &actual.re) || !same_value(&expected.im, &actual.im) {
                    return Err(format!(
                        "{label}/{ordering}/rows{rows}/slot{slot}: {actual:?} != {expected:?}"
                    ));
                }
            }
            writeln!(log, "{label},{ordering},{rows},{}", rows * 3).unwrap();
        }
    }
    Ok(())
}

fn run(directory: PathBuf, selected: Option<String>, emit: bool) -> Result<(), String> {
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let context = OneLoopExpressions::new();
    let mut emitter = RustEmitter::default();
    let mut counts = "family,root_instructions,root_slots_before,root_slots_after,unique_helpers,helper_slots_before,helper_slots_after\n".to_string();
    let mut parity = "family_domain,ordering,rows,checked_complex_outputs\n".to_string();
    let mut inputs =
        "family,point,argument,real_binary64_bits,imag_binary64_bits,real_exact,imag_exact\n"
            .to_string();
    let mut selected_names = BTreeSet::new();
    for family in [
        ScalarIntegral::A0,
        ScalarIntegral::B0,
        ScalarIntegral::DB0,
        ScalarIntegral::C0,
        ScalarIntegral::D0,
    ] {
        if selected
            .as_ref()
            .is_some_and(|name| !name.eq_ignore_ascii_case(family.name()))
        {
            continue;
        }
        selected_names.insert(family.name());
        let parameters: Vec<_> = (0..family.arity())
            .map(|i| symbolica::symbol!(format!("stack_reuse::{}_{i}", family.name())).to_atom())
            .collect();
        // Use the same transparent Laurent-series constructors as the scalar
        // adapter. Compiling calls to C0/D0 symbols would merely allocate a
        // three-call wrapper whose large body was allocated as a shared helper.
        let p = &parameters;
        let series = match family {
            ScalarIntegral::A0 => oneloop::a0(&p[0], &p[1]),
            ScalarIntegral::B0 => oneloop::b0(&p[0], &p[1], &p[2], &p[3]),
            ScalarIntegral::DB0 => oneloop::db0(&p[0], &p[1], &p[2], &p[3]),
            ScalarIntegral::C0 => context.c0([&p[0], &p[1], &p[2]], [&p[3], &p[4], &p[5]], &p[6]),
            ScalarIntegral::D0 => context.d0(
                [&p[0], &p[1], &p[2], &p[3], &p[4], &p[5]],
                [&p[6], &p[7], &p[8], &p[9]],
                &p[10],
            ),
        };
        let calls = series.into_coefficients();
        eprintln!(
            "{}: building ordinary baseline and opt-in root",
            family.name()
        );
        let baseline = context
            .evaluator(&calls, &parameters)
            .map_err(|error| error.to_string())?;
        let candidate = context
            .evaluator_with_stack_reuse(&calls, &parameters)
            .map_err(|error| error.to_string())?;
        let before = baseline.export_instructions();
        let after = candidate.export_instructions();
        let signature = provenance(&before)?;
        if signature != provenance(&after)? {
            return Err(format!(
                "{} root operation provenance changed",
                family.name()
            ));
        }
        let mut helpers_before = BTreeMap::new();
        let mut helpers_after = BTreeMap::new();
        helpers(&before, &mut helpers_before)?;
        helpers(&after, &mut helpers_after)?;
        if helpers_before != helpers_after {
            return Err(format!("{} shared function body changed", family.name()));
        }
        let helper_slots: usize = helpers_before.values().map(|entry| entry.0).sum();
        writeln!(
            counts,
            "{},{},{},{},{},{},{}",
            family.name(),
            before.instructions.len(),
            before.temporary_count,
            after.temporary_count,
            helpers_before.len(),
            helper_slots,
            helper_slots
        )
        .unwrap();
        write(
            directory.join(format!("{}-root-provenance.txt", family.name())),
            signature,
        )?;
        let points = cases(family);
        for (point, values) in points.iter().enumerate() {
            for (argument, value) in values.iter().enumerate() {
                writeln!(
                    inputs,
                    "{},{point},{argument},{:016x},{:016x},{},{}",
                    family.name(),
                    value.re.to_bits(),
                    value.im.to_bits(),
                    value.re.exact_rational().unwrap(),
                    value.im.exact_rational().unwrap()
                )
                .unwrap();
            }
        }
        verify(
            &format!("{}-f64", family.name()),
            &baseline,
            &candidate,
            &points,
            0_f64,
            &[0, 1, 3, 7, 32, 3],
            &mut parity,
        )?;
        verify(
            &format!("{}-DoubleFloat", family.name()),
            &baseline,
            &candidate,
            &points,
            DoubleFloat::default(),
            &[4, 1],
            &mut parity,
        )?;
        verify(
            &format!("{}-Float256", family.name()),
            &baseline,
            &candidate,
            &points,
            Float::new(256),
            &[4, 1],
            &mut parity,
        )?;
        if emit {
            let options = EmitOptions {
                inline_arithmetic: matches!(
                    family,
                    ScalarIntegral::A0 | ScalarIntegral::B0 | ScalarIntegral::DB0
                ),
            };
            for (label, evaluator) in [("baseline", baseline), ("candidate", candidate)] {
                let module = emitter.emit_batch(evaluator, options)?;
                write(
                    directory.join(format!("{}_{}.rs", family.name().to_lowercase(), label)),
                    module.source,
                )?;
            }
        }
        // Persist completed families even if a later family's qualification fails.
        write(directory.join("slots.csv"), &counts)?;
        write(directory.join("parity.csv"), &parity)?;
        write(directory.join("inputs.csv"), &inputs)?;
    }
    if selected_names.is_empty() {
        return Err("unknown family".into());
    }
    if emit {
        write(directory.join("constants.rs"), emitter.constants_source())?;
    }
    write(
        directory.join("semantics.txt"),
        format!(
            "families={selected_names:?}\noutputs=[finite,total_single,ir_double]\nlast_input=mu_squared\nother_inputs=invariants_and_squared_masses_in_oneloop_order\nnormalization=unmodified_oneloop_expressions_no_iregi_conversion\nallocation=Symbolica_optimize_stack_exactly_once_on_fresh_ordinary_root\nshared_helpers=unchanged\nDualizer=not_used_or_modified\nbatch_policy=Whole\nC0_D0_inputs=validation/native-policy/provenance.json_and_crates/iregi-masters/examples/native_policy_probe.rs\ncomparison=identical_ordered_instruction_provenance_plus_exact_represented_outputs_precision_and_signed_zero\ntimings=not_measured\n"
        ),
    )?;
    Ok(())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let directory = args
        .next()
        .expect("usage: stack_reuse_probe output-directory [family] [--emit]");
    let mut selected = None;
    let mut emit = false;
    for arg in args {
        if arg == "--emit" {
            emit = true;
        } else {
            assert!(
                selected.replace(arg).is_none(),
                "only one family may be selected"
            );
        }
    }
    // Main has not initialized Symbolica; only this one thread uses its state.
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(move || run(directory.into(), selected, emit))
        .unwrap()
        .join()
        .unwrap()
        .unwrap();
}
