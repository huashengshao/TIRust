//! Source-only weighted native stream. All numeric operations use Symbolica DoubleFloat.
use crate::{Answer,kernel::{Prepared,Table,MOMENTS,NATIVE}};
use serde_json::{Value,json};
use sha2::{Digest,Sha256};
// Numeric transport only; no scalar master or reduction calls.
use oneloop::NativeFloat;
use std::io::{Read,Write};
use symbolica::{domains::float::{Complex,DoubleFloat as Float,FloatLike,Real,SingleFloat},evaluate::BatchExpressionEvaluator};

pub const PROTOCOL:&str="n6-sample-n5-d6-mass-second-dd-v1";
pub const MAX_ROWS:usize=256;
pub const MAX_REQUEST:usize=16393;
pub const MAX_RESPONSE:usize=33_554_432;
pub const NUMBER_BYTES:usize=512;
const WINDOW_BATCHES:usize=32;
fn zero()->Float {Float::from(0.0)}
fn czero()->Complex<Float> {Complex::new(zero(),zero())}
fn sha(raw:&[u8])->String {format!("{:x}",Sha256::digest(raw))}
fn hex(raw:&[u8])->String {raw.iter().map(|b|format!("{b:02x}")).collect()}
fn num(v:&Float)->Answer<Value> {
    if !v.is_finite() || v.get_precision()!=106 {return Err("invalid native scalar/precision".into());}
    let text=v.exact_rational().ok_or("nonfinite rational witness")?.to_string();
    if text.len()>NUMBER_BYTES {return Err("native rational exceeds fixed serialization bound".into());}
    let limbs=v.into_inner();
    Ok(json!([text,v.get_precision(),v.negative_sign(),format!("{:016x}",limbs.hi().to_bits()),format!("{:016x}",limbs.lo().to_bits())]))
}
fn complex(v:&Complex<Float>)->Answer<Value> {Ok(json!([num(&v.re)?,num(&v.im)?]))}
fn complexes(v:&[Complex<Float>])->Answer<Vec<Value>> {v.iter().map(complex).collect()}
fn numbers(v:&[Float])->Answer<Vec<Value>> {v.iter().map(num).collect()}
fn l1(v:&Complex<Float>)->Float {v.re.norm()+v.im.norm()}
fn add_complex(a:&mut Complex<Float>,b:&Complex<Float>) {a.re+=&b.re;a.im+=&b.im;}
fn add_tracked(a:&mut Float,b:&Float,minimum:&mut u32) {
    *minimum=(*minimum).min(a.get_precision()).min(b.get_precision());*a+=b;*minimum=(*minimum).min(a.get_precision());
}
fn add_complex_tracked(a:&mut Complex<Float>,b:&Complex<Float>,minimum:&mut [u32;3]) {
    add_tracked(&mut a.re,&b.re,&mut minimum[0]);add_tracked(&mut a.im,&b.im,&mut minimum[1]);
}
fn weighted(v:&Complex<Float>,w:&Float)->Complex<Float> {Complex::new(v.re.clone()*w,v.im.clone()*w)}

pub fn read_frame<R:Read>(input:&mut R)->Answer<Option<Vec<u8>>> {
    let mut prefix=[0u8;8];let mut have=0;
    while have<8 {match input.read(&mut prefix[have..]) {
        Ok(0) if have==0=>return Ok(None),Ok(0)=>return Err("truncated length".into()),Ok(n)=>have+=n,
        Err(e) if e.kind()==std::io::ErrorKind::Interrupted=>{},Err(e)=>return Err(e.to_string())}}
    let size=usize::try_from(u64::from_le_bytes(prefix)).map_err(|_|"frame length conversion")?;
    if !(9..=MAX_REQUEST).contains(&size) {return Err("request outside fixed length bound".into());}
    let mut frame=vec![0;size];input.read_exact(&mut frame).map_err(|e|e.to_string())?;Ok(Some(frame))
}
fn write_frame<W:Write>(output:&mut W,code:u8,rows:usize,cols:usize,body:&[u8])->Answer<()> {
    let length=9usize.checked_add(body.len()).ok_or("response overflow")?;
    if length>MAX_RESPONSE {return Err("response exceeds fixed limit".into());}
    let rows=u32::try_from(rows).map_err(|_|"row header overflow")?;
    let cols=u32::try_from(cols).map_err(|_|"column header overflow")?;
    output.write_all(&(length as u64).to_le_bytes()).map_err(|e|e.to_string())?;
    output.write_all(&[code]).map_err(|e|e.to_string())?;
    output.write_all(&rows.to_le_bytes()).map_err(|e|e.to_string())?;
    output.write_all(&cols.to_le_bytes()).map_err(|e|e.to_string())?;
    output.write_all(body).map_err(|e|e.to_string())?;output.flush().map_err(|e|e.to_string())
}
fn send<W:Write>(output:&mut W,code:u8,rows:usize,cols:usize,value:&Value)->Answer<()> {
    write_frame(output,code,rows,cols,&serde_json::to_vec(value).map_err(|e|e.to_string())?)
}

const CHILD_WIDTH:usize=MOMENTS*NATIVE;
const CHILD_FIELDS:usize=4;
const PARENT_COMPONENTS:usize=3;
pub const OUTPUTS:usize=CHILD_WIDTH*(CHILD_FIELDS+PARENT_COMPONENTS);
const PARENT_OFFSET:usize=CHILD_WIDTH*CHILD_FIELDS;

struct TableRuntime {
    dynamic:Option<BatchExpressionEvaluator<Complex<Float>>>,constants:Vec<Complex<Float>>,
}
impl TableRuntime {
    fn new(table:&Table,max_rows:usize)->Answer<Self> {
        for program in [&table.dynamic,&table.constants].into_iter().flatten() {
            if program.inventory.slots.checked_mul(max_rows).ok_or("table slot overflow")?>crate::program::MAX_SLOT_VALUES {
                return Err("table exceeds fixed numeric-slot inventory".into());}
        }
        let dynamic=table.dynamic.as_ref().map(crate::program::Program::instantiate).transpose()?;
        let constants=if let Some(program)=&table.constants {
            let mut evaluator=program.instantiate()?;let point=vec![czero(),czero(),czero(),czero(),Complex::new(Float::from(1.0),zero())];
            let mut values=vec![czero();table.constant_expressions.len()];
            evaluator.evaluate_batch(&point,&mut values,1).map_err(|e|e.to_string())?;values
        }else{Vec::new()};
        if constants.iter().any(|v|!v.is_finite()){return Err("nonfinite mapped table constant".into());}
        Ok(Self{dynamic,constants})
    }
    fn values(&mut self,table:&Table,parameters:&[Complex<Float>],rows:usize)->Answer<Vec<Complex<Float>>> {
        if parameters.len()!=rows*5{return Err("table input dimensions differ".into());}
        let d=table.dynamic_expressions.len();let mut unique=vec![czero();rows*d];
        if let Some(evaluator)=&mut self.dynamic {evaluator.evaluate_batch(parameters,&mut unique,rows).map_err(|e|e.to_string())?;}
        let mut output=Vec::with_capacity(rows*table.map.len());
        for row in 0..rows {for &[kind,index]in &table.map {output.push(match kind {
            0=>czero(),1=>unique[row*d+index].clone(),2=>self.constants[index].clone(),_=>return Err("invalid table map kind".into())});}}
        if output.iter().any(|v|!v.is_finite()){return Err("nonfinite native table output".into());}Ok(output)
    }
}
struct Rule {
    index:usize,order:usize,permutation:Vec<usize>,expected_rows:usize,
    rows:usize,batches:usize,windows:usize,window_first:usize,window_rows:usize,
    weight_sum:Float,window_weight:Float,window_values:Vec<Complex<Float>>,window_abs:Vec<Float>,
    totals:Vec<Complex<Float>>,absolute:Vec<Float>,minimum:Vec<[u32;3]>,weight_min:u32,
    min_f:[Option<Float>;2],max_f:[Option<Float>;2],min_p0:Option<Float>,max_p0:Option<Float>,
    input_hash:Sha256,chain:[u8;32],window_chain:[u8;32],
}
impl Rule {
    fn new(index:usize)->Self {
        let mut seed=PROTOCOL.as_bytes().to_vec();seed.extend_from_slice(&(index as u32).to_le_bytes());let chain=Sha256::digest(seed).into();
        Self{index,order:if index==0{12}else{16},permutation:if index==0{(0..5).collect()}else{(0..5).rev().collect()},
            expected_rows:if index==0{20736}else{65536},rows:0,batches:0,windows:0,window_first:1,window_rows:0,
            weight_sum:zero(),window_weight:zero(),window_values:vec![czero();OUTPUTS],window_abs:vec![zero();OUTPUTS],
            totals:vec![czero();OUTPUTS],absolute:vec![zero();OUTPUTS],minimum:vec![[106;3];OUTPUTS],weight_min:106,
            min_f:[None,None],max_f:[None,None],min_p0:None,max_p0:None,input_hash:Sha256::new(),chain,window_chain:chain}
    }
    fn cumulative_hash(&self)->String{format!("{:x}",self.input_hash.clone().finalize())}
    fn checkpoint(&mut self)->Answer<Value> {
        if self.window_rows==0{return Ok(Value::Null);}
        for i in 0..OUTPUTS {add_complex_tracked(&mut self.totals[i],&self.window_values[i],&mut self.minimum[i]);
            add_tracked(&mut self.absolute[i],&self.window_abs[i],&mut self.minimum[i][2]);}
        add_tracked(&mut self.weight_sum,&self.window_weight,&mut self.weight_min);self.windows+=1;
        let value=json!({"window_index":self.windows,"first_batch":self.window_first,"last_batch":self.batches,
            "rows":self.window_rows,"cumulative_rows":self.rows,"weight_sum":num(&self.window_weight)?,
            "chain_before_first":hex(&self.window_chain),"chain_after_last":hex(&self.chain),"cumulative_input_sha256":self.cumulative_hash(),
            "weighted_sums":complexes(&self.window_values)?,"absolute_l1_sums":numbers(&self.window_abs)?,
            "accumulation_min_precision_bits":self.minimum,"weight_sum_global_min_precision_bits":self.weight_min});
        self.window_rows=0;self.window_first=self.batches+1;self.window_weight=zero();self.window_chain=self.chain;
        self.window_values.fill(czero());self.window_abs.fill(zero());Ok(value)
    }
}
pub struct Worker<'a> {
    prepared:&'a Prepared,unit:TableRuntime,derivatives:TableRuntime,
    geometry:[BatchExpressionEvaluator<Complex<Float>>;2],gap:BatchExpressionEvaluator<Complex<Float>>,
    rule:Option<Rule>,completed:usize,map_digest:String,
}
impl<'a>Worker<'a> {
    pub fn new(prepared:&'a Prepared)->Answer<Self> {
        if prepared.unit.map.len()!=CHILD_WIDTH||prepared.first_second_f_derivatives.map.len()!=CHILD_WIDTH*2 {
            return Err("complete native unit/first/second source tables required".into());}
        for program in prepared.geometry.iter().chain([&prepared.gap_taylor_map]) {
            if program.inventory.slots.checked_mul(MAX_ROWS).ok_or("auxiliary slot overflow")?>crate::program::MAX_SLOT_VALUES {
                return Err("geometry/gap exceeds fixed256row slot budget".into());}
        }
        let unit=TableRuntime::new(&prepared.unit,2*MAX_ROWS)?;
        let derivatives=TableRuntime::new(&prepared.first_second_f_derivatives,MAX_ROWS)?;
        let geometry=[prepared.geometry[0].instantiate()?,prepared.geometry[1].instantiate()?];let gap=prepared.gap_taylor_map.instantiate()?;
        let map_digest=sha(&serde_json::to_vec(&json!({"basis":prepared.basis,"unit":prepared.unit.map,
            "f_derivatives":prepared.first_second_f_derivatives.map})).map_err(|e|e.to_string())?);
        Ok(Self{prepared,unit,derivatives,geometry,gap,rule:None,completed:0,map_digest})
    }
    pub fn ready(&self)->Answer<Value>{Ok(json!({"protocol":PROTOCOL,"requested_topology":6,"integrated_topology":5,
        "dimension_shift_h":1,"parameters":5,"wire_stick_axes":4,"input_columns":8,"input_encoding":"nodes_weight_f64",
        "geometry_arithmetic":"Symbolica_DoubleFloat","scalar_arithmetic":"Symbolica_DoubleFloat",
        "number_encoding":"exact-rational-doublefloat-limbs-v1","working_precision_bits":106,"input_precision_bits":53,
        "precision_semantics":"fixed nominal format precision; not an achieved error bound",
        "moments":MOMENTS,"native_slots":NATIVE,"parent_components":PARENT_COMPONENTS,"outputs":OUTPUTS,
        "child_field_order":["A0","mass0_A_first","mass0_A_second_Taylor","B0"],"child_fields_width":PARENT_OFFSET,
        "parent_offset":PARENT_OFFSET,"parent_width":CHILD_WIDTH*PARENT_COMPONENTS,
        "output_order":"four child fields each moment/native, then parent moment/native/Taylor",
        "basis":self.prepared.basis,"poles":["uv1","ir1","ir2","finite"],"parent_shape":[[0,0],[1,0],[2,0]],
        "native_normalization_included":true,"ir_poles_proven_zero":true,"uv_poles_proven_zero":false,
        "positive_f_lower_bounds":["39/4","47/4"],"positive_f_upper_bounds":["24","27"],"p0_bounds":["0","1"],
        "physical_mu":"3","gap_identity":"(B0-(A0+a*A1+a²*A2))/(16-a)","parameter_relabel_B":"[5,1,2,3,4] with unchanged q",
        "native_volume":"1/24","window_batches":WINDOW_BATCHES,"max_rows":MAX_ROWS,"max_request_bytes":MAX_REQUEST,
        "max_response_bytes":MAX_RESPONSE,"frame_prefix_bytes":8,"byte_bounds_exclude_length_prefix":true,
        "absolute_sum_norm":"sum(abs(rounded(weight*real))+abs(rounded(weight*imag)))",
        "absolute_sum_semantics":"rounded L1 estimate; not a rigorous enclosure","max_number_bytes":NUMBER_BYTES,
        "source_map_sha256":self.map_digest,"kernel_artifact":self.prepared.artifact(),
        "unit_constant_values":complexes(&self.unit.constants)?,"f_derivative_constant_values":complexes(&self.derivatives.constants)?,
        "instruction_major_symbolica_batch":true,"no_scalar_master_or_reducer_calls":true,"no_jit":true,
        "accuracy_qualified":false,"default_Hybrid_qualified":false}))}
    fn start(&mut self,body:&[u8])->Answer<Value> {
        if self.rule.is_some()||self.completed>=2{return Err("rule state does not admit START".into());}
        let request:Value=serde_json::from_slice(body).map_err(|e|e.to_string())?;let rule=Rule::new(self.completed);
        if request!=json!({"configuration":rule.index,"order":rule.order,"permutation":rule.permutation}){return Err("START is not exact next fixed rule".into());}
        let response=json!({"status":"rule_started","configuration":rule.index,"order":rule.order,"permutation":rule.permutation,
            "rows":0,"batches":0,"windows":0,"batch_chain_sha256":hex(&rule.chain),"cumulative_input_sha256":rule.cumulative_hash()});
        self.rule=Some(rule);Ok(response)
    }
    fn evaluate(&mut self,payload:&[u8],rows:usize)->Answer<Value> {
        let rule=self.rule.as_mut().ok_or("EVAL without START")?;
        if rows==0||rows>MAX_ROWS||rows.checked_mul(64).and_then(|n|n.checked_add(9))!=Some(payload.len())
            ||rule.rows.checked_add(rows).ok_or("row overflow")?>rule.expected_rows{return Err("EVAL dimensions or fixed row bound differs".into());}
        let mut nodes=Vec::with_capacity(rows*5);let mut weights=Vec::with_capacity(rows);
        for row in payload[9..].chunks_exact(64) {
            let values=row.chunks_exact(8).map(|b|f64::from_le_bytes(b.try_into().expect("exact8bytechunk"))).collect::<Vec<_>>();
            if values.iter().any(|v|!v.is_finite())||values[..4].iter().any(|v|*v<=0.0||*v>=1.0)
                ||values[4..].iter().any(|w|*w<=0.0||*w>1.0){return Err("inadmissible original sticknode/axisweight".into());}
            nodes.extend(values[..4].iter().map(|v|Complex::new(Float::from(*v),zero())));nodes.push(czero());
            let mut weight=Float::from(1.0);for value in &values[4..]{weight*=Float::from(*value);}weights.push(weight);
        }
        let mut geometry=vec![czero();rows*8];self.geometry[rule.index].evaluate_batch(&nodes,&mut geometry,rows).map_err(|e|e.to_string())?;
        let mut unit_parameters=Vec::with_capacity(rows*10);let mut derivative_parameters=Vec::with_capacity(rows*5);
        for g in geometry.chunks_exact(8) {
            if g.iter().any(|v|!v.is_finite()||!v.im.is_zero())||g[0].re<zero()||g[0].re>Float::from(1.0)
                ||g[6].re<=zero()||g[6].re>=Float::from(1.0)||g[7].re<=zero()||g[7].re>=Float::from(1.0){return Err("native geometry leaves SAMPLE domain".into());}
            for child in 0..2 {
                let f=&g[4+child].re;let lower=if child==0{Float::from(39.0)/Float::from(4.0)}else{Float::from(47.0)/Float::from(4.0)};
                let upper=Float::from(if child==0{24.0}else{27.0});
                if *f<lower||*f>upper{return Err("native geometry violates fixed positive-F bounds".into());}
                if rule.min_f[child].as_ref().is_none_or(|v|f<v){rule.min_f[child]=Some(f.clone());}
                if rule.max_f[child].as_ref().is_none_or(|v|f>v){rule.max_f[child]=Some(f.clone());}
                unit_parameters.extend(g[..4].iter().cloned());unit_parameters.push(g[4+child].clone());
            }
            derivative_parameters.extend(g[..4].iter().cloned());derivative_parameters.push(g[4].clone());
            if rule.min_p0.as_ref().is_none_or(|v|g[0].re<*v){rule.min_p0=Some(g[0].re.clone());}
            if rule.max_p0.as_ref().is_none_or(|v|g[0].re>*v){rule.max_p0=Some(g[0].re.clone());}
        }
        let unit=self.unit.values(&self.prepared.unit,&unit_parameters,2*rows)?;
        let derivatives=self.derivatives.values(&self.prepared.first_second_f_derivatives,&derivative_parameters,rows)?;
        let mut all=vec![czero();rows*OUTPUTS];let mut gap_parameters=Vec::with_capacity(rows*CHILD_WIDTH*5);
        for row in 0..rows {let g=&geometry[row*8..row*8+8];for field in 0..CHILD_WIDTH {
            let a=unit[(2*row)*CHILD_WIDTH+field].clone();let b=unit[(2*row+1)*CHILD_WIDTH+field].clone();
            let first=weighted(&derivatives[row*CHILD_WIDTH*2+2*field],&g[6].re);
            let second=weighted(&derivatives[row*CHILD_WIDTH*2+2*field+1],&g[7].re);
            all[row*OUTPUTS+field]=a.clone();all[row*OUTPUTS+CHILD_WIDTH+field]=first.clone();
            all[row*OUTPUTS+2*CHILD_WIDTH+field]=second.clone();all[row*OUTPUTS+3*CHILD_WIDTH+field]=b.clone();
            gap_parameters.extend([a,first,second,b,czero()]);
        }}
        // Symbolica owns every Taylor coefficient in the exact partial-fraction map.
        for (chunk_index,chunk)in gap_parameters.chunks(MAX_ROWS*5).enumerate() {
            let count=chunk.len()/5;let mut values=vec![czero();count*3];self.gap.evaluate_batch(chunk,&mut values,count).map_err(|e|e.to_string())?;
            for (offset,value)in values.chunks_exact(3).enumerate(){let flat=chunk_index*MAX_ROWS+offset;let row=flat/CHILD_WIDTH;let field=flat%CHILD_WIDTH;
                all[row*OUTPUTS+PARENT_OFFSET+3*field..row*OUTPUTS+PARENT_OFFSET+3*field+3].clone_from_slice(value);}
        }
        if all.iter().any(|v|!v.is_finite()){return Err("nonfinite complete point component".into());}
        for (row,weight)in weights.iter().enumerate(){rule.window_weight+=weight;for field in 0..OUTPUTS {
            let value=weighted(&all[row*OUTPUTS+field],weight);add_complex(&mut rule.window_values[field],&value);rule.window_abs[field]+=l1(&value);
        }}
        rule.rows+=rows;rule.window_rows+=rows;rule.batches+=1;let input=Sha256::digest(payload);rule.input_hash.update(payload);
        let mut chain=Sha256::new();chain.update(rule.chain);chain.update((payload.len()as u64).to_le_bytes());chain.update((rows as u32).to_le_bytes());chain.update(input);rule.chain=chain.finalize().into();
        let checkpoint=if rule.batches%WINDOW_BATCHES==0{rule.checkpoint()?}else{Value::Null};
        Ok(json!({"status":"batch_accumulated","configuration":rule.index,"batch_index":rule.batches,"rows":rows,
            "cumulative_rows":rule.rows,"cumulative_batches":rule.batches,"input_payload_bytes":payload.len(),"input_sha256":hex(&input),
            "cumulative_input_sha256":rule.cumulative_hash(),"batch_chain_sha256":hex(&rule.chain),"checkpoint":checkpoint}))
    }
    fn finish(&mut self)->Answer<Value> {
        let rule=self.rule.as_mut().ok_or("FINAL without START")?;let expected_batches=rule.expected_rows/MAX_ROWS;
        if rule.rows!=rule.expected_rows||rule.batches!=expected_batches{return Err("FINAL before exact rule inventory".into());}
        let checkpoint=rule.checkpoint()?;let expected_windows=expected_batches.div_ceil(WINDOW_BATCHES);
        if rule.windows!=expected_windows{return Err("fixed checkpoint inventory differs".into());}
        let result=json!({"status":"rule_complete","configuration":rule.index,"order":rule.order,"permutation":rule.permutation,
            "rows":rule.rows,"batches":rule.batches,"windows":rule.windows,"weight_sum":num(&rule.weight_sum)?,
            "cumulative_input_sha256":rule.cumulative_hash(),"batch_chain_sha256":hex(&rule.chain),"source_map_sha256":self.map_digest,
            "weighted_sums":complexes(&rule.totals)?,"absolute_l1_sums":numbers(&rule.absolute)?,"checkpoint":checkpoint,
            "accumulation_min_precision_bits":rule.minimum,"weight_sum_global_min_precision_bits":rule.weight_min,
            "geometry_extrema":{"minimum_F_A":num(rule.min_f[0].as_ref().ok_or("missing F_A")?)?,"maximum_F_A":num(rule.max_f[0].as_ref().ok_or("missing F_A")?)?,
                "minimum_F_B":num(rule.min_f[1].as_ref().ok_or("missing F_B")?)?,"maximum_F_B":num(rule.max_f[1].as_ref().ok_or("missing F_B")?)?,
                "minimum_P0":num(rule.min_p0.as_ref().ok_or("missing P0")?)?,"maximum_P0":num(rule.max_p0.as_ref().ok_or("missing P0")?)?}});
        self.completed+=1;self.rule=None;Ok(result)
    }
    pub fn serve<R:Read,W:Write>(&mut self,input:&mut R,output:&mut W)->Answer<()> {
        send(output,2,0,0,&self.ready()?)?;
        loop {let frame=match read_frame(input){Ok(Some(v))=>v,Ok(None)if self.rule.is_none()&&self.completed==2=>return Ok(()),
            Ok(None)=>return Err("EOF before complete fixed cohort".into()),Err(e)=>{write_frame(output,1,0,0,e.as_bytes())?;return Err(e);}};
            let code=frame[0];let rows=u32::from_le_bytes(frame[1..5].try_into().map_err(|_|"row header")?)as usize;
            let cols=u32::from_le_bytes(frame[5..9].try_into().map_err(|_|"column header")?)as usize;
            let answer=(||->Answer<Option<(u8,usize,usize,Value)>>{match code {
                10 if rows==0&&cols==0=>Ok(Some((10,0,0,self.start(&frame[9..])?))),1 if cols==8=>Ok(Some((0,rows,0,self.evaluate(&frame,rows)?))),
                4 if rows==0&&cols==0&&frame.len()==9=>Ok(Some((4,0,OUTPUTS,self.finish()?))),
                2 if rows==0&&cols==0&&frame.len()==9&&self.rule.is_none()&&self.completed==2=>Ok(None),_=>Err("invalid opcode/shape/state".into())}})();
            match answer{Ok(Some((code,rows,cols,value)))=>send(output,code,rows,cols,&value)?,Ok(None)=>{write_frame(output,3,0,0,&[])?;return Ok(());},
                Err(e)=>{write_frame(output,1,0,0,e.as_bytes())?;return Err(e);}}
        }
    }
}
#[cfg(test)]mod tests {
    use super::*;use symbolica::domains::rational::Rational;
    #[test]fn wire_lift_is_exact_binary64_in_doublefloat(){
        let v=Float::from(f64::from_bits(0x3ff0000000000001));assert_eq!(v.get_precision(),106);
        assert_eq!(v.exact_rational().unwrap(),Rational::from((4503599627370497_i64,4503599627370496_i64)));
        let negative_zero=Float::from(-0.0);assert!(negative_zero.negative_sign());assert_eq!(negative_zero.exact_rational().unwrap(),Rational::from(0));
        let pi=zero().pi();assert_ne!(pi.into_inner().lo(),0.0);assert_ne!(pi.exact_rational().unwrap(),Float::from(std::f64::consts::PI).exact_rational().unwrap());
    }
}
