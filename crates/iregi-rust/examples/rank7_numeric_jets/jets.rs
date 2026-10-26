//! Exact numerical input preparation only; all algebra uses Symbolica.
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{dual::HyperDual, float::Complex, rational::{Q, Rational}},
    evaluate::{Dualizer, ExpressionEvaluator},
    tensors::matrix::Matrix,
};

pub const FIRST: [[usize; 2]; 2] = [[0, 0], [1, 0]];
pub const MIXED: [[usize; 2]; 6] = [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]];

/// m0² -> lambda²*m0²+a; all shifts and mu -> lambda times base.
/// These are independent mass and dilation directions, not two scale seeds.
fn expressions(base: &[Rational], n: usize) -> Result<(Vec<Atom>, [Atom; 2]), String> {
    if n == 0 || n.checked_mul(5).and_then(|width| width.checked_add(1)) != Some(base.len()) {
        return Err("incorrect ordinary input dimensions".into());
    }
    let a=Atom::var(symbolica::symbol!("iregi_validation::rank7_mass_seed"));
    let b=Atom::var(symbolica::symbol!("iregi_validation::rank7_dilation_seed"));
    let lambda=Atom::num(1)+&b;
    let lambda2=lambda.pow(2);
    let roots=base.iter().enumerate().map(|(i,value)| {
        let scale=if (4*n..5*n).contains(&i) {&lambda2} else {&lambda};
        let value=Atom::num(value.clone())*scale;
        if i==4*n {value+&a} else {value}
    }).collect();
    Ok((roots,[a,b]))
}

pub fn input_jets(base: &[Rational],n:usize,shape:&[Vec<usize>]) -> Result<Vec<Rational>,String> {
    let (roots,variables)=expressions(base,n)?;
    let exact:ExpressionEvaluator<Complex<Rational>>=Atom::evaluator_multiple(&roots,&variables)
        .direct_translation(true).cores(1).horner_iterations(0).build().map_err(|e|e.to_string())?;
    let mut evaluator=exact.vectorize(&Dualizer::new(HyperDual::<Complex<Rational>>::new(shape.to_vec()),vec![]))
        .map_err(|e|e.to_string())?.map_to_ring(&Q).map_err(|e|e.to_string())?;
    let components=shape.len();
    let mut seeds=vec![Rational::from(0);2*components];
    for (i,power) in shape.iter().enumerate() {
        if power==&vec![1,0] {seeds[i]=Rational::from(1);}
        if power==&vec![0,1] {seeds[components+i]=Rational::from(1);}
    }
    let mut output=vec![Rational::from(0);roots.len()*components];
    evaluator.evaluate_in_ring(&seeds,&mut output,&Q);
    if output.chunks_exact(components).zip(base).any(|(jet,value)|&jet[0]!=value) {
        return Err("input Dualizer changed the exact primal".into());
    }
    Ok(output)
}

/// Evaluate one finite point on the same nonlinear path without Dual numbers.
/// Used for an independent finite-difference derivative route.
pub fn points(base:&[Rational],n:usize,points:&[(Rational,Rational)]) -> Result<Vec<Vec<Rational>>,String> {
    let (roots,variables)=expressions(base,n)?;
    let exact:ExpressionEvaluator<Complex<Rational>>=Atom::evaluator_multiple(&roots,&variables)
        .direct_translation(true).cores(1).horner_iterations(0).build().map_err(|e|e.to_string())?;
    let mut evaluator=exact.map_to_ring(&Q).map_err(|e|e.to_string())?;
    let mut result=Vec::with_capacity(points.len());
    for (a,b) in points {
        let mut output=vec![Rational::from(0);roots.len()];
        evaluator.evaluate_in_ring(&[a.clone(),b.clone()],&mut output,&Q);
        result.push(output);
    }
    Ok(result)
}

/// Symbolica differentiates the independent homogeneity+dotted response.
/// Columns multiply ordinary unit/dot0/triple0 arrays respectively.
/// Taylor normalization is the existing physical_derivatives.rs convention.
pub fn response(n:usize,rank:usize) -> Result<Vec<Vec<Rational>>,String> {
    let a_symbol=symbolica::symbol!("iregi_validation::rank7_mass_seed");
    let b_symbol=symbolica::symbol!("iregi_validation::rank7_dilation_seed");
    let a=Atom::var(a_symbol);let b=Atom::var(b_symbol);
    let unit=Atom::var(symbolica::symbol!("iregi_validation::rank7_unit_response"));
    let dotted=Atom::var(symbolica::symbol!("iregi_validation::rank7_dot_response"));
    let tripled=Atom::var(symbolica::symbol!("iregi_validation::rank7_triple_response"));
    let lambda=Atom::num(1)+&b;let d=rank as i64+4-2*n as i64;
    let response=&unit*lambda.pow(d)+&a*&dotted*lambda.pow(d-2)+a.pow(2)*&tripled*lambda.pow(d-4);
    let da=response.derivative(a_symbol);let db=response.derivative(b_symbol);
    let roots=[response,da.clone(),db.clone(),da.derivative(a_symbol)/Atom::num(2),
               da.derivative(b_symbol),db.derivative(b_symbol)/Atom::num(2)];
    let exact:ExpressionEvaluator<Complex<Rational>>=Atom::evaluator_multiple(&roots,&[a,b,unit,dotted,tripled])
        .direct_translation(true).cores(1).horner_iterations(0).build().map_err(|e|e.to_string())?;
    let mut evaluator=exact.map_to_ring(&Q).map_err(|e|e.to_string())?;
    let mut columns=Vec::new();
    for anchor in 0..3 {
        let mut input=vec![Rational::from(0);5];input[2+anchor]=Rational::from(1);
        let mut output=vec![Rational::from(0);6];
        evaluator.evaluate_in_ring(&input,&mut output,&Q);columns.push(output);
    }
    Ok((0..6).map(|i|columns.iter().map(|column|column[i].clone()).collect()).collect())
}

/// Exact five-point Taylor weights copied from the already executed physical
/// C0/D0 derivative validation. Matrix::solve is Symbolica's solver.
pub fn stencil(order:usize) -> Result<Vec<Rational>,String> {
    if order>2 {return Err("only Taylor orders0..2 are planned".into());}
    let nodes:Vec<i64>=(-2..=2).collect();
    let matrix=Matrix::from_nested_vec((0..5).map(|degree|nodes.iter()
        .map(|x|Rational::from(x.pow(degree))).collect()).collect(),Q).map_err(|e|e.to_string())?;
    let rhs=Matrix::new_vec((0..5).map(|degree|Rational::from(i64::from(degree==order))).collect(),Q);
    Ok(matrix.solve(&rhs).map_err(|e|e.to_string())?.into_vec())
}
