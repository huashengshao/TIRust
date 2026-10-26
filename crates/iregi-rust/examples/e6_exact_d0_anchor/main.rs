//! SOURCE ONLY: one copied D0, two existing shifted Gauss--Jacobi rules.
//! No reducer/scalar-master imports, f64 input wire, or adaptive integration.
#[path = "../parameter_oracle/kernel.rs"]
mod kernel;

use std::{cmp::Ordering, fs};
use serde_json::{Value, json};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{float::{Complex, Float, FloatLike, SingleFloat}, integer::Integer, rational::Rational},
    evaluate::BatchExpressionEvaluator,
};

const BITS: u32 = 256;
const BLOCK: usize = 256;
const CAPTURED: &str = include_str!("early-D0-input.tsv");
type C = Complex<Float>;
type Answer<T> = Result<T, String>;

fn require(ok: bool, reason: &str) -> Answer<()> {
    if ok { Ok(()) } else { Err(reason.into()) }
}

fn rational(text: &str) -> Answer<Rational> {
    let (n,d) = text.split_once('/').unwrap_or((text,"1"));
    let n = n.parse::<Integer>().map_err(|e| format!("integer numerator: {e}"))?;
    let d = d.parse::<Integer>().map_err(|e| format!("integer denominator: {e}"))?;
    require(d != Integer::from(0), "zero denominator")?;
    Ok(Rational::from((n,d)))
}

fn zero() -> C { Complex::new(Float::new(BITS), Float::new(BITS)) }
fn real(value: Float) -> C { Complex::new(value, Float::new(BITS)) }
fn q(n: i64, d: i64) -> Rational { Rational::from((n,d)) }

fn component(value: &Float) -> Answer<Value> {
    require(value.is_finite(), "nonfinite output component")?;
    Ok(json!({"rational":value.try_to_rational().ok_or("finite rational conversion failed")?.to_string(),
        "precision":value.get_precision(),"negative_sign":value.is_sign_negative(),"finite":true}))
}
fn complex(value: &C) -> Answer<Value> {
    Ok(json!({"re":component(&value.re)?,"im":component(&value.im)?}))
}

fn captured() -> Answer<Vec<C>> {
    let mut values = Vec::new();
    for line in CAPTURED.lines().filter(|s| !s.trim().is_empty() && !s.starts_with('#')) {
        let fields = line.split('\t').collect::<Vec<_>>();
        require(fields.len()==3, "captured component needs rational/precision/sign")?;
        require(fields[1]=="256", "captured precision changed")?;
        let negative = match fields[2] {"true"=>true,"false"=>false,_=>return Err("explicit sign required".into())};
        let exact = rational(fields[0])?;
        let mut value = Float::new(BITS).from_rational(&exact);
        if value.is_zero() && negative { value = -value; }
        require(value.try_to_rational()==Some(exact) && value.get_precision()==BITS
            && value.is_sign_negative()==negative, "copied component changed value/precision/sign")?;
        values.push(value);
    }
    require(values.len()==22, "exactly one eleven-argument D0 row required")?;
    let input = values.chunks_exact(2).map(|v| Complex::new(v[0].clone(),v[1].clone())).collect::<Vec<_>>();
    require(input[10].re.try_to_rational()==Some(q(9,1)) && input[10].im.is_zero(), "mu squared must remain nine")?;
    // Only s02 is positive, equal to 1/4.  On the simplex, x0*x2<=1/4,
    // hence Re(F)>=min Re(m_i^2)-1/16>1; all mass widths are negative.
    for mass in &input[6..10] {
        require(mass.re > Float::new(BITS).from_rational(&q(17,16))
            && mass.im < Float::new(BITS), "copied point is outside proved positive-Re(F) domain")?;
    }
    Ok(input)
}

#[derive(Clone)]
struct Rule { nodes: Vec<Float>, weights: Vec<Float> }

fn rules(text: &str) -> Answer<Vec<Vec<Rule>>> {
    let prototype = Float::new(BITS);
    let one = prototype.one();
    let mut answer = Vec::new();
    let mut lines = text.lines().filter(|s| !s.trim().is_empty() && !s.starts_with('#'));
    for order in [20_usize,32] {
        let mut axes = Vec::new();
        for axis in 0..3 {
            let mut rule = Rule {nodes:Vec::new(),weights:Vec::new()};
            let mut weight_sum = prototype.zero();
            for index in 0..order {
                let fields = lines.next().ok_or("incomplete rule table")?.split('\t').collect::<Vec<_>>();
                require(fields.len()==5 && fields[0]==order.to_string() && fields[1]==axis.to_string()
                    && fields[2]==index.to_string(), "rule order/axis/index differs")?;
                let node = prototype.from_rational(&rational(fields[3])?);
                let weight = prototype.from_rational(&rational(fields[4])?);
                require(node.is_finite() && weight.is_finite() && node>prototype.zero() && node<one
                    && weight>prototype.zero() && rule.nodes.last().is_none_or(|last|node>*last), "invalid positive ordered rule")?;
                weight_sum += &weight;
                rule.nodes.push(node); rule.weights.push(weight);
            }
            let expected = prototype.from_rational(&q(1,(3-axis) as i64));
            let difference = weight_sum-expected.clone();
            let tolerance = expected*prototype.from_rational(&q(8,1_000_000_000_000_000));
            require(difference<=tolerance && difference>=-tolerance, "Jacobi axis normalization differs")?;
            axes.push(rule);
        }
        answer.push(axes);
    }
    require(lines.next().is_none(), "unexpected extra rule rows")?;
    Ok(answer)
}

fn evaluator(input: &[C]) -> Answer<(BatchExpressionEvaluator<C>,Vec<String>)> {
    let x: [Atom;4] = std::array::from_fn(|i|kernel::variable(&format!("anchor_x{i}")));
    let masses: [Atom;4] = std::array::from_fn(|i|kernel::variable(&format!("anchor_m{i}")));
    let route = [
        [Atom::num(0),Atom::num(0),Atom::num(0),Atom::num(0)],
        [Atom::num(0),Atom::num(0),Atom::num(0),Atom::num(1)],
        [Atom::num(q(1,2)),Atom::num(0),Atom::num(0),Atom::num(0)],
        [Atom::num(0),Atom::num(1),Atom::num(1),Atom::num(0)],
    ];
    let square = |p: &[Atom;4]|p[0].pow(2)-p[1].pow(2)-p[2].pow(2)-p[3].pow(2);
    for (index,(i,j)) in [(0,1),(1,2),(2,3),(0,3),(0,2),(1,3)].into_iter().enumerate() {
        let delta = std::array::from_fn(|mu|&route[i][mu]-&route[j][mu]);
        require(input[index].im.is_zero() && kernel::exact_zero(&(square(&delta)
            -Atom::num(input[index].re.try_to_rational().ok_or("nonfinite invariant")?))), "captured invariants do not match exact route")?;
    }
    // The existing quadrature.py formula, constructed and lowered by Symbolica.
    let p: [Atom;4] = std::array::from_fn(|mu|Atom::add_many(&(0..4).map(|i|&x[i]*&route[i][mu]).collect::<Vec<_>>()));
    let f = Atom::add_many(&(0..4).map(|i|&x[i]*(&masses[i]-square(&route[i]))).collect::<Vec<_>>())+square(&p);
    let symbols = kernel::Symbols::new();
    let [uv,finite] = kernel::direct_source_derivative(4,[0;4],&symbols)?;
    require(kernel::exact_zero(&uv) && symbols.p.iter().chain(&symbols.t).all(|s|!finite.contains(s)), "scalar kernel retained a source or momentum")?;
    let native = [uv,Atom::new(),Atom::new(),finite];
    // Proof-only checks use Symbolica, not a second integrand implementation.
    let raw_scalar = (&native[3]/kernel::native_phase()).together();
    require(kernel::exact_zero(&(&raw_scalar-symbols.f.pow(-2))), "raw scalar density is not exactly F^-2")?;
    let r=kernel::variable("sign_real_F");let s=kernel::variable("sign_imag_F");
    let imaginary=Atom::num(Complex::new(q(0,1),q(1,1)));
    let plus=&r+&imaginary*&s;let minus=&r-&imaginary*&s;
    let sign_identity=(plus.pow(-2)-minus.pow(-2))/(Atom::num(2)*&imaginary)
        +Atom::num(2)*&r*&s/(r.pow(2)+s.pow(2)).pow(2);
    require(kernel::exact_zero(&sign_identity), "positive-F imaginary-part identity failed")?;
    let mut roots = vec![f.clone()];
    for slot in &native { roots.push(kernel::substitute(slot,&[(symbols.f.clone(),f.clone())])); }
    for slot in &native {
        // The comparison target is raw OneLOop D0, not native IREGI i*pi² D0.
        // Divide once symbolically; never form binary64 pi or a numeric adapter.
        let raw = (slot/kernel::native_phase()).together();
        roots.push(kernel::substitute(&raw,&[(symbols.f.clone(),f.clone())]));
    }
    let expression_strings = roots.iter().map(|a|a.to_canonical_string()).collect();
    let parameters = x.into_iter().chain(masses).collect::<Vec<_>>();
    let exact = Atom::evaluator_multiple(&roots,&parameters).direct_translation(true).horner_iterations(0).cores(1)
        .build().map_err(|e|format!("Symbolica scalar parameter evaluator: {e}"))?;
    let prototype = Float::new(BITS);
    let mapped = exact.map_coeff_with_prec(&|r|Complex::new(prototype.from_rational(&r.re),prototype.from_rational(&r.im)),BITS);
    Ok((mapped.into_batch_evaluator(),expression_strings))
}

fn integrate(evaluator: &mut BatchExpressionEvaluator<C>, input: &[C], axes: &[Rule], order: usize) -> Answer<Value> {
    let prototype = Float::new(BITS);
    let total = order*order*order;
    let mut sums = vec![zero();8];
    let mut weight_sum = prototype.zero();
    let mut minimum_f_real: Option<Float> = None;
    let mut minimum_precision = BITS;
    let mut batches = 0;
    for start in (0..total).step_by(BLOCK) {
        let rows = BLOCK.min(total-start);
        let mut parameters = Vec::with_capacity(rows*8);
        let mut weights = Vec::with_capacity(rows);
        for ordinal in start..start+rows {
            let digits = [ordinal/(order*order),(ordinal/order)%order,ordinal%order];
            let mut remaining = prototype.one();
            let mut weight = prototype.one();
            for axis in 0..3 {
                let node = &axes[axis].nodes[digits[axis]];
                parameters.push(real(remaining.clone()*node));
                remaining *= prototype.one()-node;
                weight *= &axes[axis].weights[digits[axis]];
            }
            parameters.push(real(remaining));
            parameters.extend(input[6..10].iter().cloned());
            weights.push(weight);
        }
        let mut values = vec![zero();rows*9];
        evaluator.evaluate_batch(&parameters,&mut values,rows).map_err(|e|format!("scalar batch: {e}"))?;
        for (row,weight) in values.chunks_exact(9).zip(&weights) {
            require(row.iter().all(|z|z.re.is_finite()&&z.im.is_finite()), "nonfinite F/density")?;
            require(matches!(row[0].re.real_cmp(&prototype.one()),Some(Ordering::Equal|Ordering::Greater))
                && row[0].im<prototype.zero(), "node left proved positive-Re(F), negative-width domain")?;
            if minimum_f_real.as_ref().is_none_or(|old|row[0].re<*old) { minimum_f_real=Some(row[0].re.clone()); }
            for slot in [1,2,3,5,6,7] { require(row[slot].is_zero(), "scalar UV/IR slot is not exact zero")?; }
            require(row[8].im>prototype.zero(), "raw F^-2 imaginary part violates the positive-F/negative-width sign proof")?;
            for slot in 0..8 {
                sums[slot] += &row[slot+1]*weight;
                minimum_precision=minimum_precision.min(row[slot+1].re.get_precision()).min(row[slot+1].im.get_precision());
            }
            weight_sum += weight;
        }
        batches += 1;
    }
    let native = sums[..4].iter().map(complex).collect::<Answer<Vec<_>>>()?;
    let raw = sums[4..].iter().map(complex).collect::<Answer<Vec<_>>>()?;
    Ok(json!({"order":order,"points":total,"batches":batches,"native_slots":native,"oneloop_raw_slots":raw,
        "weight_sum":component(&weight_sum)?,"minimum_real_F":component(&minimum_f_real.ok_or("empty rule")?)?,
        "minimum_density_component_precision":minimum_precision,"native_scalar_poles_exact_zero":true,"raw_finite_positive_imaginary_at_all_nodes":true,
        "quadrature_error_bound_certified":false,"higher_precision_D0_qualified":false}))
}

fn run() -> Answer<Value> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    require(args.len()==2, "usage: worker COPIED_INPUT.tsv SCIPY_RULES.tsv")?;
    require(fs::read_to_string(&args[0]).map_err(|e|e.to_string())?==CAPTURED, "runtime input differs from embedded exact captured TSV")?;
    let input = captured()?;
    let axes = rules(&fs::read_to_string(&args[1]).map_err(|e|e.to_string())?)?;
    let (mut evaluator,expressions) = evaluator(&input)?;
    let results = axes.iter().zip([20_usize,32]).map(|(a,n)|integrate(&mut evaluator,&input,a,n)).collect::<Answer<Vec<_>>>()?;
    Ok(json!({"status":"two_fixed_rule_values_completed_not_error_certified","topology":4,"rank":0,"dimension_shift":0,
        "powers":[1,1,1,1],"working_precision":BITS,"captured_input":input.iter().map(complex).collect::<Answer<Vec<_>>>()?,
        "sign_identity_proved_by_Symbolica":true,"raw_scalar_density_verified_F_inverse_squared":true,"input_common_t_reconstructed":false,"input_f64_conversions":0,"laurent_order":["uv1","ir1","ir2","finite"],
        "routing":[["0","0","0","0"],["0","0","0","1"],["1/2","0","0","0"],["0","1","1","0"]],
        "native_normalization":"unchanged kernel; i*pi^2; mu energy 3; Gamma(N) included once",
        "oneloop_raw_conversion":"each kernel slot divided symbolically by kernel::native_phase() once",
        "F_domain_proof":"only positive pair is s02=1/4; x0*x2<=1/4; min(Re m2)>17/16; Im masses<0",
        "rule_source":"SciPy roots_sh_jacobi(n,3,1),(n,2,1),(n,1,1); represented binary64 rules rationalized exactly",
        "rule_precision_is_not_physical_input_precision":true,"jacobian_already_in_rule_weights":true,
        "expressions":["F","native UV1","native IR1","native IR2","native finite","raw UV1","raw IR1","raw IR2","raw finite"],
        "symbolica_expressions":expressions,"results":results,"independent_D0_agreement_qualified":false,
        "performance_evidence":false,"Dual_qualified":false,"E6_or_full_N6_qualified":false}))
}

fn main() {
    match run() {
        Ok(value)=>println!("{}",serde_json::to_string_pretty(&value).expect("JSON value serializes")),
        Err(error)=>{eprintln!("exact D0 parameter anchor: {error}");std::process::exit(1);}
    }
}
