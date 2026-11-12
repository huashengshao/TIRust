//! S-wave projected numerators of the g g > c c~ diagrams (benchmarks/onia,
//! physical amplitudes).
//!
//! Every DOT file of DIAGRAM_DIR (examples/onia_diagrams.rs) gets the
//! projector of the 1S0 colour-singlet state at p_c = p_c~ = P/2: the open
//! c line u-bar_i ... v_j is replaced by
//!   Pi_ji = [(pslash_c~ - m) gamma5 (pslash_c + m)]_ji / sqrt(8 m^3),
//! the colours by delta_ij / sqrt(3) and delta^ab / sqrt(8) (STATE `singlet`)
//! or by sqrt(2) T^c_ji d^abc / sqrt(40/3), summed over c (STATE `octet`);
//! the gluon polarizations stay symbolic and are bound per point. The
//! numerator is prepared with the model's Feynman rules and evaluated at the
//! onia lattice points (points.json, process gg_QQ) for two real linear
//! polarizations of each gluon and the gauge vector k / k^0 (index 2, for the
//! Ward identity; never for both gluons at once); G = 1 (the amplitude is
//! proportional to G^4 at one loop).
//!
//! usage: onia_swave MODEL_JSON DIAGRAM_DIR POINTS_JSON OUTPUT_JSON [MAX_POINTS [STATE]]
//! Output: {"state", "diagrams": [{"name", "loops", "loop_propagators": [{"loop",
//! "external", "pdg"}], "tree_propagators": [...], "external_legs":
//! [{"edge", "index", "incoming"}], "external_edges": [...], "points": [{"index",
//! "m2", "physical_momenta", "polarization_vectors": [gluon 1's three, gluon 2's three],
//! "polarizations": [[l1, l2, {"masses_squared": [..], "tree_masses_squared": [..],
//! "terms": [[[e0,e1,e2,e3], [re, im]]...]}]...]}]}]}
use std::{collections::HashMap, error::Error};

use feynkit_graph::symbols;
use iregi_prepare::{NumeratorSource, PrepareOptions, prepare_dot_with_tree_propagators};
use serde_json::{Value, json};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::float::{Complex, Float},
    function, symbol,
};

const BIS: &str = "spenso::{spenso::representation,spenso::self_dual}::bis";
const MINK: &str = "spenso::{spenso::representation,spenso::self_dual}::mink";
const COAD: &str = "spenso::{spenso::representation,spenso::self_dual}::coad";
const COF: &str = "spenso::{spenso::representation,spenso::dualizable}::cof";
const DIND: &str = "spenso::{spenso::lower}::dind";
const GAMMA: &str = "spenso::{linear,spenso::tensor}::gamma";
const METRIC: &str = "spenso::{symmetric,linear,real,spenso::tensor}::g";
const Q: &str = "gammalooprs::{spenso::rank1,spenso::tensor,spenso::component-labels:subscript}::Q";
const EPS: &str = "gammalooprs::{spenso::rank1,spenso::tensor}::ϵ";
const GENERATOR: &str = "spenso::{real,spenso::tensor}::t";

fn hedge(edge: usize) -> String {
    format!("gammalooprs::{{spenso::index}}::hedge({edge},1)")
}

/// The 1S0 projector, for c on edge 2 and c~ on edge 3 and the gluons on
/// edges 0 and 1 (onia_diagrams.rs's external order).
///
/// Colour: FeynKit writes the physical generator element T^a_ij (i the c, j
/// the c~ index) as t(a, cof(j), dind(cof(i))), so the open c line's colour
/// X_ij is contracted with P(cof(i), dind(cof(j))) = P_ij. Singlet: P_ij =
/// delta_ij / sqrt(3), times delta^ab / sqrt(8) for the gluons. Octet:
/// sum_c sqrt(2) T^c_ji d^abc / sqrt(40/3) = sqrt(3/20) ({T^a, T^b}_ji -
/// delta_ij delta^ab / 3) (Fierz); d^abc's normalization sum_abc d^2 = 40/3
/// makes the c sum the colour-summed amplitude.
fn swave_projector(state: &str) -> Result<String, Box<dyn Error>> {
    let (h0, h1, h2, h3) = (hedge(0), hedge(1), hedge(2), hedge(3));
    let bis = |x: &str| format!("{BIS}(4,{x})");
    let mink = |x: &str| format!("{MINK}(4,{x})");
    let slash = |row: &str, column: &str, edge: usize, mu: &str| {
        format!("{GAMMA}({},{},{})*{Q}({edge},{})", bis(row), bis(column), mink(mu), mink(mu))
    };
    let one = |row: &str, column: &str| format!("{METRIC}({},{})", bis(row), bis(column));
    let spin = format!(
        "(({}) - UFO::MC*{})*spenso::gamma5({},{})*(({}) + UFO::MC*{})/(8*UFO::MC^3)^(1/2)",
        slash(&h3, "onia::a", 3, "onia::mu1"),
        one(&h3, "onia::a"),
        bis("onia::a"),
        bis("onia::b"),
        slash("onia::b", &h2, 2, "onia::mu2"),
        one("onia::b", &h2),
    );
    let delta = format!("{METRIC}({COF}(3,{h2}),{DIND}({COF}(3,{h3})))");
    let gluons = format!("{METRIC}({COAD}(8,{h0}),{COAD}(8,{h1}))");
    let colour = match state {
        "singlet" => format!("{delta}/3^(1/2)*{gluons}/8^(1/2)"),
        "octet" => {
            // (T^a T^b)_ji = T^a_jk T^b_ki = t(a, cof(k), dind(j)) t(b, cof(i), dind(k)).
            let product = |a: &str, b: &str| {
                format!(
                    "{GENERATOR}({COAD}(8,{a}),{COF}(3,onia::k),{DIND}({COF}(3,{h3})))*{GENERATOR}({COAD}(8,{b}),{COF}(3,{h2}),{DIND}({COF}(3,onia::k)))"
                )
            };
            format!("(3/20)^(1/2)*({}+{}-{delta}*{gluons}/3)", product(&h0, &h1), product(&h1, &h0))
        }
        other => return Err(format!("unknown state {other} (singlet or octet)").into()),
    };
    let polarizations = format!("{EPS}(0,{})*{EPS}(1,{})", mink(&h0), mink(&h1));
    Ok(format!("{spin}*{colour}*{polarizations}"))
}

fn with_projector(dot: &str, projector: &str) -> Result<String, Box<dyn Error>> {
    let start = dot.find("projector=\"").ok_or("no projector attribute")? + "projector=\"".len();
    let end = start + dot[start..].find('"').ok_or("unterminated projector")?;
    Ok(format!("{}{}{}", &dot[..start], projector, &dot[end..]))
}

fn number(value: &Value) -> f64 {
    value.as_str().map_or_else(|| value.as_f64().unwrap(), |text| text.parse().unwrap())
}

/// Two real unit polarization vectors orthogonal to the gluon's direction,
/// then the gauge vector k / k^0.
fn polarizations(k: &[f64; 4]) -> [[f64; 4]; 3] {
    let n = [k[1] / k[0], k[2] / k[0], k[3] / k[0]];
    let trial = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let dot = trial[0] * n[0] + trial[1] * n[1] + trial[2] * n[2];
    let mut e1 = [trial[0] - dot * n[0], trial[1] - dot * n[1], trial[2] - dot * n[2]];
    let norm = (e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]).sqrt();
    e1.iter_mut().for_each(|x| *x /= norm);
    let e2 = [n[1] * e1[2] - n[2] * e1[1], n[2] * e1[0] - n[0] * e1[2], n[0] * e1[1] - n[1] * e1[0]];
    [[0.0, e1[0], e1[1], e1[2]], [0.0, e2[0], e2[1], e2[2]], [1.0, n[0], n[1], n[2]]]
}

fn evaluate(atom: &Atom, bindings: &HashMap<Atom, Complex<Float>>) -> Result<[f64; 2], Box<dyn Error>> {
    let value = atom.evaluate_with_prec(bindings, 128).map_err(|e| format!("{e}: {atom}"))?.to_f64();
    Ok([value.re, value.im])
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model = std::fs::read_to_string(&args[1])?;
    let points: Value = serde_json::from_str(&std::fs::read_to_string(&args[3])?)?;
    let max_points: usize = args.get(5).map_or(Ok(usize::MAX), |s| s.parse())?;
    let state = args.get(6).map_or("singlet", String::as_str);
    let process = points["processes"].as_array().unwrap().iter().find(|p| p["id"] == "gg_QQ").ok_or("gg_QQ")?;
    let mut names: Vec<_> = std::fs::read_dir(&args[2])?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "dot"))
        .collect();
    names.sort();
    let projector = swave_projector(state)?;
    let options = PrepareOptions { source: NumeratorSource::FeynmanRules, external_wavefunctions: false, component: None };
    let mut diagrams = Vec::new();
    for path in names {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let dot = with_projector(&std::fs::read_to_string(&path)?, &projector)?;
        let loops = if name.contains("-L1-") { 1 } else { 0 };
        let prepared = if loops == 1 {
            prepare_dot_with_tree_propagators(&model, &dot, &options).map_err(|e| format!("{name}: {e}"))?
        } else {
            // Tree diagrams: one-loop only in prepare_dot; their projected
            // numerators come from the Born evaluation in swave.py.
            diagrams.push(json!({"name": name, "loops": 0}));
            continue;
        };
        let mut point_out = Vec::new();
        for point in process["points"].as_array().unwrap().iter().take(max_points) {
            let momenta = &point["momenta"];
            let vector = |leg: &str, sign: f64| -> [f64; 4] {
                std::array::from_fn(|mu| sign * number(&momenta[leg][mu]))
            };
            // Physical momenta: incoming gluons, outgoing c and c~ (= -incoming Q).
            let physical = [vector("g1", 1.0), vector("g2", 1.0), vector("Q", -1.0), vector("Qb", -1.0)];
            let m2 = number(&point["m2"]);
            let mut base: HashMap<Atom, Complex<f64>> = HashMap::new();
            base.insert(Atom::var(symbol!("UFO::ZERO")), Complex::new(0.0, 0.0));
            base.insert(Atom::var(symbol!("UFO::MC")), Complex::new(m2.sqrt(), 0.0));
            base.insert(Atom::var(symbol!("UFO::G")), Complex::new(1.0, 0.0));
            // Massless light quarks (u, d, s loops).
            for light in ["UFO::MU", "UFO::MD", "UFO::MS"] {
                base.insert(Atom::var(symbol!(light)), Complex::new(0.0, 0.0));
            }
            for (slot, edge) in prepared.external_edges.iter().enumerate() {
                let leg = prepared.external_legs.iter().find(|leg| leg.edge_id == *edge).unwrap();
                for mu in 0..4 {
                    let component = function!(symbol!("spenso::cind"), Atom::num(mu as i64));
                    base.insert(
                        function!(symbols::external_momentum(), Atom::num(slot as i64), component),
                        Complex::new(physical[leg.index][mu], 0.0),
                    );
                }
            }
            let gluon_pols = [polarizations(&physical[0]), polarizations(&physical[1])];
            let mut pols = Vec::new();
            for l1 in 0..3 {
                for l2 in 0..3 {
                    if l1 == 2 && l2 == 2 {
                        continue;
                    }
                    let mut bindings = base.clone();
                    for (edge, pol) in [(0usize, gluon_pols[0][l1]), (1, gluon_pols[1][l2])] {
                        for mu in 0..4 {
                            let component = function!(symbol!("spenso::cind"), Atom::num(mu as i64));
                            bindings.insert(
                                function!(symbols::epsilon(), Atom::num(edge as i64), component),
                                Complex::new(pol[mu], 0.0),
                            );
                        }
                    }
                    let precise: HashMap<_, _> = bindings
                        .iter()
                        .map(|(a, v)| (a.clone(), Complex::new(Float::with_val(128, v.re), Float::with_val(128, v.im))))
                        .collect();
                    let terms = prepared
                        .numerator
                        .terms
                        .iter()
                        .map(|t| Ok(json!([t.exponents, evaluate(&t.coefficient, &precise)?])))
                        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
                    let masses = prepared.propagators.iter().map(|p| evaluate(&p.mass_squared, &precise).map(|v| v[0])).collect::<Result<Vec<_>, _>>()?;
                    let tree = prepared.tree_propagators.iter().map(|p| evaluate(&p.mass_squared, &precise).map(|v| v[0])).collect::<Result<Vec<_>, _>>()?;
                    pols.push(json!([l1, l2, {"masses_squared": masses, "tree_masses_squared": tree, "terms": terms}]));
                }
            }
            point_out.push(json!({"index": point["index"], "m2": point["m2"], "polarizations": pols,
                "physical_momenta": physical, "polarization_vectors": gluon_pols}));
        }
        let routing = |p: &iregi_prepare::PropagatorRouting| json!({"loop": p.loop_coefficient, "external": p.external_coefficients, "pdg": p.particle_pdg});
        diagrams.push(json!({
            "name": name, "loops": 1,
            "loop_propagators": prepared.propagators.iter().map(routing).collect::<Vec<_>>(),
            "tree_propagators": prepared.tree_propagators.iter().map(routing).collect::<Vec<_>>(),
            "external_edges": prepared.external_edges,
            "external_legs": prepared.external_legs.iter().map(|l| json!({"edge": l.edge_id, "index": l.index, "incoming": l.incoming})).collect::<Vec<_>>(),
            "rank": prepared.numerator.rank(),
            "points": point_out,
        }));
        eprintln!("{name}: {} loop + {} tree propagators, rank {}", prepared.propagators.len(), prepared.tree_propagators.len(), prepared.numerator.rank());
    }
    std::fs::write(&args[4], serde_json::to_string(&json!({"state": state, "diagrams": diagrams}))?)?;
    Ok(())
}
