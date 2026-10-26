//! Independent, dimensionally regulated positive-Re(F) parameter moments.
//! Every Gamma/Laurent/source-polynomial operation belongs to Symbolica.
use serde_json::{Value, json};
use symbolica::{
    atom::{Atom, AtomCore, Symbol},
    domains::{atom::AtomField, float::Complex, integer::Integer, rational::Rational},
    evaluate::ExpressionEvaluator,
    id::Replacement,
};

pub type Answer<T> = Result<T, String>;
pub const MOMENTS: usize = 330;
pub const OUTPUTS: usize = MOMENTS * 4;
pub const PARAMETERS: usize = 5;

#[derive(Clone)]
pub struct Symbols {
    pub t: [Atom; 4], pub t_symbols: [Symbol; 4],
    pub p: [Atom; 4], pub f: Atom,
    pub lambda: Atom, pub lambda_symbol: Symbol,
    pub epsilon: Atom, pub epsilon_symbol: Symbol,
    pub ell: Atom,
}
impl Symbols {
    pub fn new() -> Self {
        let t_symbols = ["t0", "tx", "ty", "tz"].map(|name|
            symbolica::symbol!(format!("low_n_parameter_oracle::{name}")));
        let lambda_symbol = symbolica::symbol!("low_n_parameter_oracle::lambda");
        let epsilon_symbol = symbolica::symbol!("low_n_parameter_oracle::epsilon");
        Self {t: t_symbols.map(Atom::var), t_symbols,
            p: ["P0", "Px", "Py", "Pz"].map(variable), f: variable("F"),
            lambda: Atom::var(lambda_symbol), lambda_symbol,
            epsilon: Atom::var(epsilon_symbol), epsilon_symbol, ell: variable("ell")}
    }
    pub fn parameters(&self) -> Vec<Atom> {
        self.p.iter().cloned().chain([self.f.clone()]).collect()
    }
}
pub fn variable(name: &str) -> Atom {
    Atom::var(symbolica::symbol!(format!("low_n_parameter_oracle::{name}")))
}
pub fn substitute(value: &Atom, pairs: &[(Atom, Atom)]) -> Atom {
    value.replace_multiple(pairs.iter().map(|(a,b)| Replacement::new(a.clone(),b.clone())))
}
pub fn exact_zero(value: &Atom) -> bool { value.expand().together().is_zero() }
pub fn native_phase() -> Atom {symbolica::parse!("𝑖") * Atom::var(Symbol::PI).pow(2)}
pub fn physical_log(s: &Symbols) -> Atom {
    Symbol::LOG.call((Atom::num(9),)) - Symbol::LOG.call((Atom::var(Symbol::PI),))
        - Symbol::LOG.call((&s.f,))
}
pub fn basis() -> Vec<[u16;4]> {
    let mut answer=Vec::new();
    for degree in 0_u16..=7 {for t in (0..=degree).rev() {for x in (0..=degree-t).rev() {
        for y in (0..=degree-t-x).rev() {answer.push([t,x,y,degree-t-x-y]);}
    }}}
    answer
}
pub fn validate_domain(n: usize) -> Answer<()> {
    if !(2..=4).contains(&n) {return Err("Laurent oracle admits ordinary N2/N3/N4, rank7, h0 only".into());}
    Ok(())
}

/// The previously source-reviewed Gaussian source kernel, with the radial
/// Gamma kept as Symbolica's built-in function. No local Gamma/Wick recurrence.
/// Gamma(N) from parametrization is already included. Native i*pi^2 is applied
/// exactly once, outside the epsilon-dependent mu^(2eps)*pi^(-eps) factor.
pub fn generating_kernel(n: usize,s: &Symbols,bookkeeping: bool) -> Answer<Atom> {
    validate_domain(n)?;
    let source: [Atom;4]=std::array::from_fn(|i|
        if bookkeeping {&s.lambda*&s.t[i]} else {s.t[i].clone()});
    let linear=Atom::add_many(&(0..4).map(|i|&source[i]*&s.p[i]).collect::<Vec<_>>());
    let metric=source[0].pow(2)-source[1].pow(2)-source[2].pow(2)-source[3].pow(2);
    let radial=Atom::add_many(&(0_u32..=3).map(|j| {
        symbolica::transcendental::gamma().call((&s.epsilon+Atom::num(n as i64-2-i64::from(j)),))
            / Atom::num(Integer::factorial(j))
            * (-&metric/Atom::num(4)).pow(i64::from(j))
            * s.f.pow(i64::from(j)+2-n as i64)
    }).collect::<Vec<_>>());
    Ok(native_phase()*Atom::num(if n%2==0 {1} else {-1})
        * Symbol::EXP.call((-linear,))*radial
        * Symbol::EXP.call((&s.epsilon*&s.ell,)))
}

/// Return UV1 and finite generating functions. F has positive real part; all
/// epsilon poles of this massive parameter representation are ultraviolet.
/// IR1/IR2 zeros are a domain theorem, not inferred from cancellation.
pub fn laurent_kernels(n:usize,s:&Symbols,bookkeeping:bool)->Answer<[Atom;2]> {
    let series=generating_kernel(n,s,bookkeeping)?.series(s.epsilon_symbol,0,0)
        .map_err(|e|format!("Symbolica Gamma Laurent preparation: {e}"))?;
    let higher=series.coefficient((-2).into()).ok_or("unknown double Laurent pole")?;
    if !exact_zero(&higher) {return Err("unexpected double pole in positive-Re(F) massive kernel".into());}
    let uv=series.coefficient((-1).into()).ok_or("unknown UV Laurent coefficient")?;
    let finite=series.coefficient(0.into()).ok_or("unknown finite Laurent coefficient")?;
    let replacement=[(s.ell.clone(),physical_log(s))];
    let result=[substitute(&uv,&replacement),substitute(&finite,&replacement)];
    if result.iter().any(|a|a.contains(&s.epsilon)||a.contains(&s.ell)) {
        return Err("Laurent preparation retained epsilon or formal logarithm".into());
    }
    Ok(result)
}

pub fn source_coefficients(expression:&Atom,s:&Symbols)->Answer<(Vec<Atom>,Atom)> {
    let truncated=expression.series(s.lambda_symbol,0,7)
        .map_err(|e|format!("Symbolica source series: {e}"))?.to_atom();
    let source_polynomial=substitute(&truncated,&[(s.lambda.clone(),Atom::num(1))]).expand();
    let field=AtomField {statistical_zero_test:false,cancel_check_on_division:true,custom_normalization:None};
    let polynomial=source_polynomial.to_polynomial_in_vars_with_field::<u32>(s.t.as_slice(),&field);
    if polynomial.variables().len()!=4 || polynomial.variables().iter().zip(&s.t).any(|(a,b)|a.to_atom()!=*b) {
        return Err("source polynomial changed requested variable order".into());
    }
    for term in &polynomial {
        if term.exponents.iter().sum::<u32>()>7 || s.t.iter().any(|t|term.coefficient.contains(t)) {
            return Err("truncated source coefficient retained source dependence".into());
        }
    }
    let values=basis().iter().map(|alpha| {
        let coefficient=polynomial.coefficient(&alpha.map(u32::from)).unwrap_or_else(Atom::new);
        coefficient*alpha.iter().map(|&power|Atom::num(Integer::factorial(u32::from(power)))).product::<Atom>()
    }).collect();
    Ok((values,source_polynomial))
}

pub fn direct_source_derivative(n:usize,alpha:[u16;4],s:&Symbols)->Answer<[Atom;2]> {
    let mut expressions=laurent_kernels(n,s,false)?;
    for expression in &mut expressions {
        for (axis,&power) in alpha.iter().enumerate() {for _ in 0..power {
            *expression=expression.derivative(s.t_symbols[axis]);
        }}
        *expression=substitute(expression,&s.t.iter().cloned().map(|t|(t,Atom::num(0))).collect::<Vec<_>>()).expand().together();
    }
    Ok(expressions)
}

