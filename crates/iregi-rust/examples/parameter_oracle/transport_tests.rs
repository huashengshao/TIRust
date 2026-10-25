//! Transport/domain tests only. Independent symbolic integral identities belong
//! to the parent oracle's mathematical tests.
use super::*;
use std::{io::Cursor, sync::{Arc, OnceLock}};
use symbolica::{atom::Atom, domains::float::RealLike,
    evaluate::{ExportedConstantFunction, ExportedSubEvaluator}};

fn prepared() -> &'static Prepared {
    static PREPARED: OnceLock<Prepared> = OnceLock::new();
    PREPARED.get_or_init(|| Prepared::new(2).expect("private N2 preparation"))
}
fn row(p0: f64, px: f64, f: Complex<f64>) -> Vec<Complex<f64>> {
    vec![Complex::new(p0,0.),Complex::new(px,0.),Complex::new(-0.,0.),
        Complex::new(0.125,0.),f]
}
fn framed(payload: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    write_frame(&mut result,payload).unwrap();
    result
}
fn assert_failure_frames(bytes: Vec<u8>) {
    let mut input = Cursor::new(bytes);
    let ready = read_frame(&mut input,MAX_RESPONSE).unwrap().unwrap();
    assert_eq!(ready[0],RESPONSE_READY);
    let failure = read_frame(&mut input,MAX_RESPONSE).unwrap().unwrap();
    assert_eq!(failure[0],RESPONSE_ERROR);
    assert_eq!(&failure[1..9],&[0;8]);
    assert!(!failure[9..].is_empty());
    assert!(decode_values_response(&failure,0).is_err());
    assert!(read_frame(&mut input,MAX_RESPONSE).unwrap().is_none());
}

#[test]
fn complex_roundtrip_preserves_empty_odd_reordered_and_chunked_rows() {
    for n in [2,3,4] { roundtrip_family(n); }
}
fn roundtrip_family(n: usize) {
    let prepared = Prepared::new(n).unwrap();
    let mut workspace = Workspace::new(&prepared).unwrap();
    let a = row(-0.,0.25,Complex::new(2.,-0.125));
    let b = row(0.5,-0.25,Complex::new(1.,0.));
    let c = row(-0.75,0.125,Complex::new(3.,-0.5));
    let input = [a.clone(),b.clone(),c.clone()].concat();
    let input_bits = complex_bytes(&input);
    let expected = workspace.evaluate(&input,3).unwrap();
    assert_eq!(complex_bytes(&input),input_bits);
    assert_eq!(expected.len(),3*OUTPUTS);
    // Independent scalar controls use the existing numeric Complex/Real APIs.
    // These check the native phase and F dependence for every admitted family.
    let pi = 0.0_f64.pi();
    let native_phase = Complex::new(0.,pi*pi);
    for (lane,point) in [&a,&b,&c].into_iter().enumerate() {
        let (slot,scalar) = match n {
            2 => (0,native_phase),
            3 => (3,-native_phase/point[4]),
            4 => (3,native_phase/point[4].pow(2)),
            _ => unreachable!("roundtrip catalog contains N2/N3/N4 only"),
        };
        let actual = expected[lane*OUTPUTS+slot];
        assert!((actual-scalar).norm().re <= 1e-13*(1.+scalar.norm().re),
            "N{n} lane {lane} scalar native slot {slot}: {actual} != {scalar}");
    }
    for (i,point) in [&a,&b,&c].into_iter().enumerate() {
        assert_eq!(workspace.evaluate(point,1).unwrap(),expected[i*OUTPUTS..(i+1)*OUTPUTS]);
    }
    let reordered = [c.clone(),a.clone(),b.clone()].concat();
    let actual = workspace.evaluate(&reordered,3).unwrap();
    for (destination,source) in [2,0,1].into_iter().enumerate() {
        assert_eq!(&actual[destination*OUTPUTS..(destination+1)*OUTPUTS],
            &expected[source*OUTPUTS..(source+1)*OUTPUTS]);
    }
    assert!(workspace.evaluate(&[],0).unwrap().is_empty());
    let stream = [framed(&request_values(0,&[]).unwrap()),
        framed(&request_values(3,&input).unwrap()),
        framed(&message(OP_QUIT,0,0,&[]).unwrap())].concat();
    let mut responses = Vec::new();
    serve(&mut workspace,n,&mut Cursor::new(stream),&mut responses).unwrap();
    let mut responses = Cursor::new(responses);
    let ready = read_frame(&mut responses,MAX_RESPONSE).unwrap().unwrap();
    assert_eq!(&ready[..9],&message(RESPONSE_READY,0,0,&[]).unwrap());
    let ready: Value = serde_json::from_slice(&ready[9..]).unwrap();
    for (key,value) in [("protocol",json!(PROTOCOL)),("topology",json!(n)),
        ("requested_topology",json!(n)),("parameters",json!(5)),
        ("moments",json!(330)),("native_slots",json!(4)),("outputs",json!(1320)),
        ("complex_layout",json!("interleaved_real_imaginary")),
        ("output_order",json!("moment_major_native_uv1_ir1_ir2_finite")),
        ("native_normalization_included",json!(true)),("ir_poles_proven_zero",json!(true)),
        ("instruction_major_symbolica_batch",json!(true)),("no_jit",json!(true))] {
        assert_eq!(ready[key],value,"READY field {key}");
    }
    let empty = read_frame(&mut responses,MAX_RESPONSE).unwrap().unwrap();
    assert!(decode_values_response(&empty,0).unwrap().is_empty());
    let values = read_frame(&mut responses,MAX_RESPONSE).unwrap().unwrap();
    assert_eq!(decode_values_response(&values,3).unwrap(),expected);
    assert_eq!(read_frame(&mut responses,MAX_RESPONSE).unwrap().unwrap(),
        message(RESPONSE_QUIT,0,0,&[]).unwrap());
    assert!(read_frame(&mut responses,MAX_RESPONSE).unwrap().is_none());
    assert_eq!(workspace.rows,12);
    assert_eq!(workspace.batches,8);
    assert_eq!(workspace.scheduled_instruction_lane_visits,workspace.instructions as u64*12);
    assert_eq!(MAX_REQUEST,20_489);
    assert_eq!(MAX_RESPONSE,5_406_729);
}

#[test]
fn malformed_or_nonfinite_requests_return_error_without_success_counters() {
    let good = row(0.,0.,Complex::new(2.,-0.25));
    let mut invalid = Vec::new();
    for (index,value) in [(0,Complex::new(f64::INFINITY,0.)),
        (1,Complex::new(0.,f64::NAN)),(2,Complex::new(0.,1.)),
        (4,Complex::new(0.999,0.)),(4,Complex::new(2.,0.001)),
        (4,Complex::new(2.,f64::NEG_INFINITY))] {
        let mut bad = good.clone(); bad[index] = value;
        invalid.push(request_values(1,&bad).unwrap());
    }
    invalid.extend([
        message(OP_EVALUATE,1,4,&[0;80]).unwrap(),
        message(OP_EVALUATE,1,PARAMETERS,&[0;79]).unwrap(),
        message(OP_EVALUATE,0,PARAMETERS,&[0]).unwrap(),
        message(99,0,0,&[]).unwrap(),
        message(OP_QUIT,1,0,&[]).unwrap(),
        message(OP_EVALUATE,MAX_ROWS+1,PARAMETERS,&[]).unwrap(),
    ]);
    let mut workspace = Workspace::new(prepared()).unwrap();
    for payload in invalid {
        let mut response = Vec::new();
        assert!(serve(&mut workspace,2,&mut Cursor::new(framed(&payload)),&mut response).is_err());
        assert_failure_frames(response);
        assert_eq!((workspace.rows,workspace.batches,workspace.scheduled_instruction_lane_visits),(0,0,0));
    }
    // One invalid lane rejects the entire batch, even when a prior lane is valid.
    let mut mixed = [good.clone(),good.clone(),good.clone()].concat();
    mixed[PARAMETERS+4].re = 0.;
    assert!(workspace.evaluate(&mixed,3).is_err());
    assert_eq!((workspace.rows,workspace.batches),(0,0));
    let mut overflowing = [good.clone(),good.clone(),good.clone()].concat();
    overflowing[PARAMETERS].re = f64::MAX;
    assert!(workspace.evaluate(&overflowing,3).is_err());
    assert_eq!((workspace.rows,workspace.batches,workspace.scheduled_instruction_lane_visits),(0,0,0));
    assert!(workspace.evaluate(&good,1).is_ok());
    let counters = (workspace.rows,workspace.batches,workspace.scheduled_instruction_lane_visits);
    assert!(workspace.evaluate(&[],usize::MAX).is_err());
    assert!(workspace.evaluate(&[],workspace.max_rows+1).is_err());
    assert!(workspace.evaluate(&good[..4],1).is_err());
    assert_eq!((workspace.rows,workspace.batches,workspace.scheduled_instruction_lane_visits),counters);
    let mut fresh = Workspace::new(prepared()).unwrap();
    assert_eq!((fresh.rows,fresh.batches,fresh.scheduled_instruction_lane_visits),(0,0,0));
    fresh.batches = u64::MAX;
    assert!(fresh.evaluate(&good,1).is_err());
    assert_eq!(fresh.rows,0);
}

#[test]
fn framing_and_client_decoder_reject_null_partial_oversize_and_wrong_responses() {
    assert!(read_frame(&mut Cursor::new(Vec::<u8>::new()),MAX_REQUEST).unwrap().is_none());
    assert!(read_frame(&mut Cursor::new(vec![1;7]),MAX_REQUEST).is_err());
    for size in [0,8,MAX_REQUEST as u64+1,u64::MAX] {
        assert!(read_frame(&mut Cursor::new(size.to_le_bytes()),MAX_REQUEST).is_err());
    }
    let valid = request_values(1,&row(0.,0.,Complex::new(2.,0.))).unwrap();
    let maximum = request_values(MAX_ROWS,&row(0.,0.,Complex::new(2.,0.)).repeat(MAX_ROWS)).unwrap();
    assert_eq!(maximum.len(),MAX_REQUEST);
    assert_eq!(read_frame(&mut Cursor::new(framed(&maximum)),MAX_REQUEST).unwrap().unwrap(),maximum);
    let mut partial = framed(&valid); partial.pop();
    assert!(read_frame(&mut Cursor::new(partial),MAX_REQUEST).is_err());
    assert!(request_values(MAX_ROWS+1,&[]).is_err());
    assert!(request_values(1,&[]).is_err());
    assert!(message(RESPONSE_VALUES,usize::MAX,OUTPUTS,&[]).is_err());
    assert!(write_frame(&mut Vec::new(),&[]).is_err());
    for payload in [vec![],vec![0;8],message(RESPONSE_VALUES,0,0,&[]).unwrap(),
        message(RESPONSE_ERROR,0,0,b"bad").unwrap(),message(RESPONSE_QUIT,0,0,&[]).unwrap(),
        message(RESPONSE_READY,0,0,b"null").unwrap(),
        message(RESPONSE_VALUES,0,OUTPUTS,&[0]).unwrap()] {
        assert!(decode_values_response(&payload,0).is_err());
    }
    assert!(decode_values_response(&message(RESPONSE_VALUES,0,OUTPUTS,&[]).unwrap(),1).is_err());
    let mut values = vec![Complex::new(0.,0.);OUTPUTS];
    values[OUTPUTS-1].im = f64::NAN;
    assert!(decode_values_response(&message(RESPONSE_VALUES,1,OUTPUTS,
        &complex_bytes(&values)).unwrap(),1).is_err());
    let mut workspace = Workspace::new(prepared()).unwrap();
    let mut response = Vec::new();
    assert!(serve(&mut workspace,2,&mut Cursor::new(vec![0;7]),&mut response).is_err());
    assert_failure_frames(response);
    assert_eq!((workspace.rows,workspace.batches),(0,0));
    let mut mismatch_output = Vec::new();
    assert!(serve(&mut workspace,3,&mut Cursor::new(vec![]),&mut mismatch_output).is_err());
    assert!(mismatch_output.is_empty());
}

#[test]
fn evaluator_audit_retains_function_slot_and_memory_guards() {
    let exported = prepared().exact_evaluator().unwrap().export_instructions();
    let inventory = audit(&exported).unwrap();
    assert!(inventory.slots >= PARAMETERS+OUTPUTS);
    assert!(inventory.dynamic_logarithms > 0);
    let mut invalid = exported.clone(); invalid.input_count -= 1;
    assert!(audit(&invalid).is_err());
    let mut invalid = exported.clone(); invalid.output_count -= 1;
    assert!(audit(&invalid).is_err());
    let mut invalid = exported.clone(); invalid.temporary_count = usize::MAX;
    assert!(audit(&invalid).is_err());
    let mut invalid = exported.clone(); invalid.temporary_count = MAX_SLOT_VALUES;
    assert!(audit(&invalid).is_err());
    let mut invalid = exported.clone();
    invalid.sub_evaluators.push(ExportedSubEvaluator {
        symbol:Symbol::LOG,tags:vec![],input_count:1,output_count:1,
        instructions:Arc::new(exported.clone()) });
    assert!(audit(&invalid).is_err());
    for instruction in [
        Instruction::Fun(Slot::Out(0),Box::new((Symbol::SIN,vec![],vec![Slot::Param(4)])),false),
        Instruction::Fun(Slot::Out(0),Box::new((Symbol::LOG,vec!["tag".into()],vec![Slot::Param(4)])),false),
        Instruction::Fun(Slot::Out(0),Box::new((Symbol::LOG,vec![],vec![])),false),
        Instruction::Fun(Slot::Out(0),Box::new((Symbol::LOG,vec![],vec![Slot::Param(0)])),false),
        Instruction::Assign(Slot::Out(0),Slot::Temp(usize::MAX)),
        Instruction::Assign(Slot::Const(0),Slot::Param(0)),
        Instruction::Assign(Slot::Out(usize::MAX),Slot::Param(0)),
        Instruction::Powf(Slot::Out(0),Slot::Param(4),Slot::Param(0),false),
        Instruction::IfElse(Slot::Param(0),0),
    ] {
        let mut invalid = exported.clone(); invalid.instructions.insert(0,instruction);
        assert!(audit(&invalid).is_err());
    }
    for definition in [
        ExportedConstantFunction { index:exported.constants.len(),symbol:Symbol::PI,tags:vec![],fixed_args:vec![] },
        ExportedConstantFunction { index:0,symbol:Symbol::SIN,tags:vec![],fixed_args:vec![] },
        ExportedConstantFunction { index:0,symbol:Symbol::LOG,tags:vec![],fixed_args:vec![Complex::new(Rational::from(-1),Rational::from(0))] },
    ] {
        let mut invalid = exported.clone(); invalid.constant_functions.push(definition);
        assert!(audit(&invalid).is_err());
    }
    let mut invalid = exported.clone(); invalid.instructions.clear();
    assert!(audit(&invalid).is_err());
}

#[test]
fn float_workspace_resolves_constants_at_precision_and_preserves_beyond_f64_inputs() {
    let prepared = prepared();
    let exact = prepared.exact_evaluator().unwrap();
    let exported = exact.export_instructions();
    assert!(exported.constant_functions.iter().any(|constant| constant.symbol == Symbol::PI));
    assert!(exported.constant_functions.iter().any(|constant|
        constant.symbol == symbolica::transcendental::euler_gamma()));
    for bits in [128,256] {
        let prototype = Float::new(bits);
        let mapped = exact.clone().map_coeff_with_prec(&|value|Complex::new(
            prototype.from_rational(&value.re),prototype.from_rational(&value.im)),bits);
        let mapped_constants = mapped.export_instructions().constants;
        for definition in &exported.constant_functions {
            let expected = if definition.symbol == Symbol::PI { prototype.pi() }
                else if definition.symbol == symbolica::transcendental::euler_gamma() { prototype.euler() }
                else { prototype.from_rational(&definition.fixed_args[0].re).log() };
            assert_eq!(mapped_constants[definition.index].re,expected);
            assert!(mapped_constants[definition.index].im.is_zero());
        }
        let mut workspace = FloatWorkspace::with_precision(prepared,prototype.clone()).unwrap();
        let make = |rational|Complex::new(prototype.from_rational(&rational),prototype.zero());
        let a = vec![make(Rational::from((1,8))),make(Rational::from((1,4))),
            make(Rational::from((1,2))),make(Rational::from((3,4))),
            Complex::new(prototype.from_i64(2),prototype.from_rational(&Rational::from((-1,8))))];
        let mut b = a.clone();
        b[4].re += prototype.from_rational(&Rational::from((1_u64,1_u64<<60)));
        assert_ne!(a[4].re,b[4].re);
        assert_eq!(a[4].re.to_f64(),b[4].re.to_f64());
        let input = [a.clone(),b.clone(),a.clone()].concat();
        let output = workspace.evaluate(&input,3).unwrap();
        assert_eq!(&output[..OUTPUTS],&output[2*OUTPUTS..]);
        assert!(!(output[3].clone()-&output[OUTPUTS+3]).is_zero());
        assert_eq!(workspace.evaluate(&a,1).unwrap(),output[..OUTPUTS]);
        assert_eq!(workspace.evaluate(&b,1).unwrap(),output[OUTPUTS..2*OUTPUTS]);
        assert!(workspace.evaluate(&[],0).unwrap().is_empty());
        assert_eq!(workspace.ready(2)["working_precision_bits"],bits);
    }
    // Exact IR atoms are an admission condition, not inferred from a result array.
    let mut invalid = Prepared::new(2).unwrap(); invalid.expressions[1] = Atom::num(1);
    assert!(Workspace::new(&invalid).is_err());
}
