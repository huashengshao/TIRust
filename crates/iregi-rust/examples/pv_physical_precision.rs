//! Independent precision probe for the canonical physical pentagon.
//! Decimal inputs enter through Symbolica Float at 1024 bits, then its exact
//! binary-rational conversion; no input is rounded through binary64.
#[path = "../src/basis.rs"]
mod basis;
#[path = "../src/ibp.rs"]
mod ibp;
#[path = "../src/pv.rs"]
mod pv;
use oneloop::{NativeEvaluator, ScalarIntegral};
use std::{collections::HashMap, path::PathBuf, time::Instant};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        float::{Complex, Float, SingleFloat},
        rational::Rational,
    },
    evaluate::ExpressionEvaluator,
};
fn constant(s: &str) -> Atom {
    let value = Float::parse(s, Some(1024)).unwrap();
    Atom::num(value.try_to_rational().expect("finite decimal input"))
}
fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let input_path = std::env::var("IREGI_PV_PHYSICAL_INPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root.join("validation/physical/gghhh-cli-input.json"));
    let input: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&input_path).unwrap()).unwrap();
    let propagators = input["propagators"].as_array().unwrap();
    assert!(
        input
            .get("dimension_shift")
            .is_none_or(|v| v.as_i64() == Some(0)),
        "probe requires dimension shift zero"
    );
    assert!(
        propagators
            .iter()
            .all(|p| p.get("power").is_none_or(|v| v.as_u64() == Some(1))),
        "probe requires unit propagator powers"
    );
    assert!(
        input.get("scalar_products").is_none(),
        "probe requires coordinate-derived invariants"
    );
    let shifts: Vec<[Atom; 4]> = propagators
        .iter()
        .map(|p| std::array::from_fn(|mu| constant(p["shift"][mu].as_str().unwrap())))
        .collect();
    let masses: Vec<_> = propagators
        .iter()
        .map(|p| {
            assert!(
                constant(p["mass_squared"][1].as_str().unwrap()).is_zero(),
                "probe requires real squared masses"
            );
            constant(p["mass_squared"][0].as_str().unwrap())
        })
        .collect();
    let scale = constant(input["mu"].as_str().unwrap()).pow(2);
    let all_basis = basis::tensor_basis(input["rank"].as_u64().unwrap() as u16);
    let selected_spec = std::env::var("IREGI_PV_PHYSICAL_MOMENTS").unwrap_or_else(|_| "54".into());
    let selected: Vec<usize> = if selected_spec == "all" {
        (0..all_basis.len()).collect()
    } else {
        selected_spec
            .split(',')
            .map(|x| x.parse().unwrap())
            .collect()
    };
    assert!(selected.iter().all(|&index| index < all_basis.len()));
    let precisions: Vec<u32> = std::env::var("IREGI_PV_PHYSICAL_BITS")
        .unwrap_or_else(|_| "128,256".into())
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    let started = Instant::now();
    let alias_mode = std::env::var("IREGI_PV_PHYSICAL_ALIASES").is_ok_and(|s| s == "1");
    eprintln!("physical PV exact preparation: moments {selected:?}");
    let mut compiler = if alias_mode {
        pv::PvCompiler::with_aliases(shifts.clone(), masses.clone())
    } else {
        pv::PvCompiler::new(shifts.clone(), masses.clone())
    }
    .unwrap();
    let expressions: Vec<_> = selected
        .iter()
        .map(|&i| compiler.compile_moment(all_basis[i].0).unwrap())
        .collect();
    eprintln!(
        "physical PV prepared in {:?}, {} master leaves",
        started.elapsed(),
        compiler.leaves().len()
    );
    let alias_evaluator: Option<ExpressionEvaluator<Complex<Rational>>> = if alias_mode {
        let parameters: Vec<_> = compiler
            .leaves()
            .iter()
            .flat_map(|leaf| leaf.slots.iter().cloned())
            .collect();
        let roots: Vec<_> = expressions.iter().flatten().cloned().collect();
        eprintln!(
            "physical PV lowering {} exact aliases at {:?}",
            compiler.aliases().len(),
            started.elapsed()
        );
        let evaluator = Atom::evaluator_multiple(&roots, &parameters)
            .add_aliases(compiler.aliases().iter().cloned())
            .unwrap()
            .direct_translation(true)
            .horner_iterations(0)
            .cores(1)
            .build()
            .unwrap();
        eprintln!(
            "physical PV exact aliases lowered at {:?}",
            started.elapsed()
        );
        Some(evaluator)
    } else {
        None
    };
    let empty: HashMap<Atom, Float> = HashMap::new();
    let mut outputs = Vec::new();
    for bits in precisions {
        let zero = Float::new(bits);
        let c = |value: Float| Complex::new(value, Float::new(bits));
        let number = |a: &Atom| a.evaluate_with_prec(&empty, bits).unwrap();
        let mut evaluators: Vec<_> = [
            ScalarIntegral::A0,
            ScalarIntegral::B0,
            ScalarIntegral::C0,
            ScalarIntegral::D0,
        ]
        .map(|family| NativeEvaluator::<Float>::with_binary_precision(family, bits).unwrap())
        .into_iter()
        .collect();
        let mut values: [HashMap<Atom, Float>; 2] = std::array::from_fn(|_| HashMap::new());
        let mut master_values = Vec::new();
        for leaf in compiler.leaves() {
            let active = &leaf.active;
            let mut args = Vec::new();
            let mut distance = |i: usize, j: usize| {
                let d: [Atom; 4] =
                    std::array::from_fn(|mu| &shifts[active[i]][mu] - &shifts[active[j]][mu]);
                let s = &d[0] * &d[0] - &d[1] * &d[1] - &d[2] * &d[2] - &d[3] * &d[3];
                args.push(c(number(&s)));
            };
            match active.len() {
                1 => {}
                2 => distance(0, 1),
                3 => {
                    distance(0, 1);
                    distance(1, 2);
                    distance(0, 2);
                }
                4 => {
                    distance(0, 1);
                    distance(1, 2);
                    distance(2, 3);
                    distance(0, 3);
                    distance(0, 2);
                    distance(1, 3);
                }
                _ => unreachable!(),
            }
            args.extend(active.iter().map(|&i| c(number(&masses[i]))));
            args.push(c(number(&scale)));
            let mut result = vec![c(zero.clone()); 3];
            evaluators[active.len() - 1]
                .evaluate(&args, &mut result)
                .unwrap();
            let uv = match active.len() {
                1 => c(number(&masses[active[0]])),
                2 => c(Float::with_val(bits, 1)),
                _ => c(zero.clone()),
            };
            for (slot, value) in leaf.slots.iter().zip([
                uv.clone(),
                result[1].clone() - uv,
                result[2].clone(),
                result[0].clone(),
            ]) {
                master_values.push(value.clone());
                values[0].insert(slot.clone(), value.re);
                values[1].insert(slot.clone(), value.im);
            }
        }
        let alias_outputs = alias_evaluator.as_ref().map(|exact| {
            let mut evaluator = exact.clone().map_coeff_with_prec(
                &|a| Complex::new(zero.from_rational(&a.re), zero.from_rational(&a.im)),
                bits,
            );
            let mut output = vec![c(zero.clone()); expressions.len() * 4];
            evaluator.evaluate(&master_values, &mut output);
            output
        });
        let mut moments = Vec::new();
        for (position, (&index, expression)) in selected.iter().zip(&expressions).enumerate() {
            let slots: Vec<_> = if let Some(output) = &alias_outputs {
                output[position * 4..(position + 1) * 4]
                    .iter()
                    .map(|value| [value.re.to_string(), value.im.to_string()])
                    .collect()
            } else {
                let components: [Vec<_>; 2] = std::array::from_fn(|component| {
                    expression
                        .iter()
                        .map(|e| e.evaluate_with_prec(&values[component], bits).unwrap())
                        .collect()
                });
                (0..4)
                    .map(|slot| {
                        [
                            components[0][slot].to_string(),
                            components[1][slot].to_string(),
                        ]
                    })
                    .collect()
            };
            moments.push(serde_json::json!({"index":index,"monomial":all_basis[index].0,"oneloop_native_slots":slots}));
        }
        eprintln!(
            "physical PV precision {bits} finished at {:?}",
            started.elapsed()
        );
        outputs.push(serde_json::json!({"bits":bits,"moments":moments}));
    }
    println!("{}",serde_json::to_string_pretty(&serde_json::json!({"input":input_path,"input_snapshot":input,"dimension_shift":0,"propagator_powers":"unit","input_decimal_embedding_bits":1024,"method":"exact Symbolica PV coefficients and native oneloop scalar masters","representation":if alias_mode {"alias_dag"} else {"inline"},"normalization":"OneLoop; UV, IR single, IR double, finite; each slot real,imag strings","results":outputs})).unwrap());
}
