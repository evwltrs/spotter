//! ONNX files read, changed and written back, so a downloaded model can be made right for DirectML
//! without Python. The file format's types are in proto.rs.

#[allow(clippy::all)]
mod proto;

use anyhow::{Context, Result, bail};
use prost::Message;
use prost::bytes::Bytes;
use proto::tensor_shape_proto::dimension::Value as Dim;
use proto::{GraphProto, ModelProto, NodeProto, TensorProto, type_proto};
use std::collections::{HashMap, HashSet};
use std::path::{Component, Path};

const INT64: i32 = 7;
const EXTERNAL: i32 = 1;

/// Read a model. Weights kept in a file beside it stay there, as references.
pub fn load(path: &Path) -> Result<ModelProto> {
	let bytes = Bytes::from(std::fs::read(path).with_context(|| format!("couldn't read {}", path.display()))?);
	ModelProto::decode(bytes).with_context(|| format!("{} isn't an ONNX model", path.display()))
}

/// Write a model, by way of a `.part` beside it so a half-written one is never taken for the model.
/// Weights kept outside the model it was read from (in the folder `from`) are copied into a file
/// beside the new one, `<file>_data`, a tensor at a time, so a big model is never all in memory.
pub fn save(m: &mut ModelProto, path: &Path, from: &Path) -> Result<()> {
	let name = path.file_name().context("a model with no file name")?.to_string_lossy().into_owned();
	let data_name = format!("{name}_data");
	let mut data: Option<std::io::BufWriter<std::fs::File>> = None;
	let mut at = 0u64;
	let mut sources: HashMap<String, std::fs::File> = HashMap::new();
	let mut copy = |t: &mut TensorProto| -> Result<()> {
		if t.data_location != Some(EXTERNAL) {
			return Ok(());
		}
		let entry =
			|k: &str| t.external_data.iter().find(|e| e.key.as_deref() == Some(k)).and_then(|e| e.value.clone());
		let file = entry("location").context("a tensor kept outside the model without saying where")?;
		if Path::new(&file).components().any(|c| !matches!(c, Component::Normal(_))) {
			bail!("{file} is outside the model's folder");
		}
		let offset: u64 = entry("offset").map(|s| s.parse()).transpose()?.unwrap_or(0);
		let src = match sources.entry(file.clone()) {
			std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
			std::collections::hash_map::Entry::Vacant(e) => {
				e.insert(std::fs::File::open(from.join(&file)).with_context(|| format!("couldn't read {file}"))?)
			}
		};
		let length: u64 = match entry("length") {
			Some(l) => l.parse()?,
			None => src.metadata()?.len().saturating_sub(offset),
		};
		std::io::Seek::seek(src, std::io::SeekFrom::Start(offset))?;
		let out = match &mut data {
			Some(d) => d,
			None => data.insert(std::io::BufWriter::new(std::fs::File::create(path.with_file_name(&data_name))?)),
		};
		if std::io::copy(&mut std::io::Read::take(&mut *src, length), out)? != length {
			bail!("{file} is shorter than the model says");
		}
		let kv = |k: &str, v: String| proto::StringStringEntryProto { key: Some(k.into()), value: Some(v) };
		t.external_data =
			vec![kv("location", data_name.clone()), kv("offset", at.to_string()), kv("length", length.to_string())];
		at += length;
		Ok(())
	};
	if let Some(g) = &mut m.graph {
		in_graph(g, &mut copy)?;
	}
	for f in &mut m.functions {
		in_nodes(&mut f.node, &mut copy)?;
	}
	if let Some(mut d) = data {
		std::io::Write::flush(&mut d)?;
	}
	let bytes = m.encode_to_vec();
	// protobuf's own limit, past which ONNX Runtime can't read it
	if bytes.len() >= 1 << 31 {
		bail!("{} would be over 2 GB in one file", path.display());
	}
	let part = crate::paths::part(path);
	std::fs::write(&part, bytes).with_context(|| format!("couldn't write {}", part.display()))?;
	std::fs::rename(&part, path)?;
	Ok(())
}

/// Every tensor in a graph and the graphs inside it.
fn in_graph(g: &mut GraphProto, f: &mut impl FnMut(&mut TensorProto) -> Result<()>) -> Result<()> {
	for t in &mut g.initializer {
		f(t)?;
	}
	for s in &mut g.sparse_initializer {
		for t in s.values.iter_mut().chain(s.indices.iter_mut()) {
			f(t)?;
		}
	}
	in_nodes(&mut g.node, f)
}

fn in_nodes(nodes: &mut [NodeProto], f: &mut impl FnMut(&mut TensorProto) -> Result<()>) -> Result<()> {
	for a in nodes.iter_mut().flat_map(|n| &mut n.attribute) {
		for t in a.t.iter_mut().chain(&mut a.tensors) {
			f(t)?;
		}
		for g in a.g.iter_mut().chain(&mut a.graphs) {
			in_graph(g, f)?;
		}
	}
	Ok(())
}

/// What `patch` changed.
#[derive(Debug, Default)]
pub struct Patched {
	/// matrix products made 2-D
	pub matmuls: usize,
	/// Reshapes with `allowzero` turned off
	pub reshapes: usize,
	/// matrix products left alone because the model doesn't say how many dimensions an input has
	pub unknown: Vec<String>,
}

/// Rewrite a model so DirectML gets it right, from one file to another.
pub fn patch(from: &Path, to: &Path) -> Result<Patched> {
	let mut m = load(from)?;
	let p = patch_model(&mut m).with_context(|| format!("couldn't fix {} for DirectML", from.display()))?;
	save(&mut m, to, from.parent().unwrap_or(Path::new(".")))?;
	Ok(p)
}

/// The names given to the varying sizes of a model's inputs, with the sizes `sizes` fixes them at
/// (as `name:1,3,1024,1024`, several joined by spaces).
pub fn named_sizes(m: &ModelProto, sizes: &str) -> Result<Vec<(String, i64)>> {
	let g = m.graph.as_ref().context("a model with no graph")?;
	let mut out = Vec::new();
	for s in sizes.split_whitespace() {
		let (name, dims) = s.split_once(':').with_context(|| format!("{s} isn't name:1,3,64,64"))?;
		let dims: Vec<i64> = dims.split(',').map(str::parse).collect::<Result<_, _>>()?;
		let v = g.input.iter().find(|v| v.name.as_deref() == Some(name)).with_context(|| format!("no input {name}"))?;
		let Some(type_proto::Value::TensorType(t)) = v.r#type.as_ref().and_then(|t| t.value.as_ref()) else {
			bail!("{name} isn't a tensor");
		};
		let have = t.shape.as_ref().map_or(&[][..], |s| &s.dim[..]);
		if have.len() != dims.len() {
			bail!("{name} has {} dimensions, not {}", have.len(), dims.len());
		}
		for (d, &n) in have.iter().zip(&dims) {
			match &d.value {
				Some(Dim::DimParam(p)) if !p.is_empty() => out.push((p.clone(), n)),
				Some(Dim::DimValue(v)) if *v != n => bail!("{name} is fixed at {v} where {n} was asked for"),
				_ => {}
			}
		}
	}
	Ok(out)
}

/// DirectML gets a matrix product with a plain list wrong: it becomes a 2-D product and a Squeeze ([K]
/// on the left is [1,K], squeezed at -2; on the right [K,1], at -1). It refuses Reshape's `allowzero`,
/// which changes nothing with a constant shape without zeros, so there it's turned off.
pub fn patch_model(m: &mut ModelProto) -> Result<Patched> {
	let opset = m.opset_import.iter().find(|o| matches!(o.domain.as_deref(), None | Some("" | "ai.onnx")));
	let opset = opset.and_then(|o| o.version).unwrap_or(0);
	let g = m.graph.as_mut().context("a model with no graph")?;
	let mut init: HashMap<String, TensorProto> =
		g.initializer.iter().map(|t| (t.name.clone().unwrap_or_default(), t.clone())).collect();
	let consts: HashMap<String, TensorProto> = g
		.node
		.iter()
		.filter(|n| n.op_type.as_deref() == Some("Constant") && !n.output.is_empty())
		.filter_map(|n| {
			let t = n.attribute.iter().find(|a| a.name.as_deref() == Some("value"))?.t.clone()?;
			Some((n.output[0].clone(), t))
		})
		.collect();
	let mut ranks: HashMap<String, Option<usize>> = HashMap::new();
	for v in g.value_info.iter().chain(&g.input).chain(&g.output) {
		let r = match v.r#type.as_ref().and_then(|t| t.value.as_ref()) {
			Some(type_proto::Value::TensorType(t)) => t.shape.as_ref().map(|s| s.dim.len()),
			_ => None,
		};
		ranks.insert(v.name.clone().unwrap_or_default(), r);
	}
	let rank = |name: &str, init: &HashMap<String, TensorProto>| -> Option<usize> {
		init.get(name).or_else(|| consts.get(name)).map(|t| t.dims.len()).or_else(|| ranks.get(name).copied().flatten())
	};

	let mut done = Patched::default();
	let mut added: Vec<TensorProto> = Vec::new();
	let mut axes: HashMap<i64, String> = HashMap::new();
	let mut axes_for = |v: i64, added: &mut Vec<TensorProto>| -> String {
		axes.entry(v)
			.or_insert_with(|| {
				let name = format!("patch1d_axes_{}{}", if v < 0 { "m" } else { "" }, v.abs());
				added.push(TensorProto {
					dims: vec![1],
					data_type: Some(INT64),
					name: Some(name.clone()),
					raw_data: Some(Bytes::copy_from_slice(&v.to_le_bytes())),
					..Default::default()
				});
				name
			})
			.clone()
	};
	let node = |op: &str, input: Vec<String>, output: Vec<String>, name: String| NodeProto {
		input,
		output,
		name: Some(name),
		op_type: Some(op.into()),
		..Default::default()
	};

	let mut nodes = Vec::with_capacity(g.node.len());
	for n in std::mem::take(&mut g.node) {
		if n.op_type.as_deref() != Some("MatMul") || n.input.len() != 2 || n.output.is_empty() {
			nodes.push(n);
			continue;
		}
		let name = n.name.clone().unwrap_or_default();
		let (ra, rb) = (rank(&n.input[0], &init), rank(&n.input[1], &init));
		if ra.is_none() || rb.is_none() {
			done.unknown.push(name.clone());
		}
		if ra != Some(1) && rb != Some(1) {
			nodes.push(n);
			continue;
		}
		if ra == rb {
			bail!("{name}: a product of two plain lists, which this doesn't rewrite");
		}
		if opset < 13 {
			bail!("{name} needs rewriting, and the model's opset ({opset}) is older than 13");
		}
		let mut inputs = n.input.clone();
		let mut squeeze = 0;
		for side in 0..2 {
			if [ra, rb][side] != Some(1) {
				continue;
			}
			let was = &n.input[side];
			let made = format!("{was}_patch1d_{}", if side == 0 { "row" } else { "col" });
			match init.get(was).or_else(|| consts.get(was)) {
				Some(t) => {
					if !init.contains_key(&made) {
						let mut t = t.clone();
						let k = t.dims.first().copied().unwrap_or(1);
						t.dims = if side == 0 { vec![1, k] } else { vec![k, 1] };
						t.name = Some(made.clone());
						added.push(t.clone());
						init.insert(made.clone(), t);
					}
				}
				None => {
					let at = axes_for(side as i64, &mut added);
					nodes.push(node(
						"Unsqueeze",
						vec![was.clone(), at],
						vec![made.clone()],
						format!("{name}_patch1d_unsq{side}"),
					));
				}
			}
			inputs[side] = made;
			squeeze = if side == 0 { -2 } else { -1 };
		}
		let y = n.output[0].clone();
		let flat = format!("{y}_patch1d_2d");
		nodes.push(node("MatMul", inputs, vec![flat.clone()], name.clone()));
		let at = axes_for(squeeze, &mut added);
		nodes.push(node("Squeeze", vec![flat, at], vec![y], format!("{name}_patch1d_squeeze")));
		done.matmuls += 1;
	}
	g.node = nodes;
	g.initializer.extend(added);

	for n in &mut g.node {
		if n.op_type.as_deref() != Some("Reshape") || n.input.len() < 2 {
			continue;
		}
		let Some(shape) = init.get(&n.input[1]).or_else(|| consts.get(&n.input[1])).and_then(int64s) else { continue };
		if shape.contains(&0) {
			continue;
		}
		for a in &mut n.attribute {
			if a.name.as_deref() == Some("allowzero") && a.i.unwrap_or(0) != 0 {
				a.i = Some(0);
				done.reshapes += 1;
			}
		}
	}

	// the 1-D weights that were reshaped into new ones go, with anything else nobody reads
	let mut used: HashSet<String> = g.output.iter().filter_map(|o| o.name.clone()).collect();
	used.extend(g.input.iter().filter_map(|o| o.name.clone()));
	inputs_of(&g.node, &mut used);
	g.initializer.retain(|t| t.name.as_ref().is_some_and(|n| used.contains(n)));
	Ok(done)
}

/// Every name the nodes read, in the graphs inside them too.
fn inputs_of(nodes: &[NodeProto], used: &mut HashSet<String>) {
	for n in nodes {
		used.extend(n.input.iter().cloned());
		for a in &n.attribute {
			for g in a.g.iter().chain(&a.graphs) {
				inputs_of(&g.node, used);
			}
		}
	}
}

/// A whole-number tensor's values.
fn int64s(t: &TensorProto) -> Option<Vec<i64>> {
	if t.data_type != Some(INT64) || t.data_location == Some(EXTERNAL) {
		return None;
	}
	Some(match &t.raw_data {
		Some(b) => b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
		None => t.int64_data.clone(),
	})
}

#[cfg(test)]
mod tests {
	use super::*;
	use proto::{AttributeProto, OperatorSetIdProto, TypeProto, ValueInfoProto};

	fn value(name: &str, dims: &[i64]) -> ValueInfoProto {
		let dim = dims
			.iter()
			.map(|&d| proto::tensor_shape_proto::Dimension { value: Some(Dim::DimValue(d)), ..Default::default() })
			.collect();
		ValueInfoProto {
			name: Some(name.into()),
			r#type: Some(TypeProto {
				value: Some(type_proto::Value::TensorType(type_proto::Tensor {
					elem_type: Some(1),
					shape: Some(proto::TensorShapeProto { dim }),
				})),
				..Default::default()
			}),
			..Default::default()
		}
	}

	fn node(op: &str, input: &[&str], output: &str) -> NodeProto {
		NodeProto {
			input: input.iter().map(|s| s.to_string()).collect(),
			output: vec![output.into()],
			name: Some(format!("{op}_{output}")),
			op_type: Some(op.into()),
			..Default::default()
		}
	}

	fn int64(name: &str, v: &[i64]) -> TensorProto {
		TensorProto {
			dims: vec![v.len() as i64],
			data_type: Some(INT64),
			name: Some(name.into()),
			raw_data: Some(Bytes::from(v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())),
			..Default::default()
		}
	}

	#[test]
	fn plain_lists_become_matrices_and_allowzero_goes() {
		let w = TensorProto {
			dims: vec![4],
			data_type: Some(1),
			name: Some("w".into()),
			float_data: vec![1.0; 4],
			..Default::default()
		};
		let mut reshape = node("Reshape", &["y", "shape"], "z");
		reshape.attribute.push(AttributeProto {
			name: Some("allowzero".into()),
			i: Some(1),
			r#type: Some(2),
			..Default::default()
		});
		let mut m = ModelProto {
			opset_import: vec![OperatorSetIdProto { domain: Some(String::new()), version: Some(17) }],
			graph: Some(GraphProto {
				// a weight on the right, then a list made inside on the left
				node: vec![node("MatMul", &["x", "w"], "y"), node("MatMul", &["v", "x"], "u"), reshape],
				initializer: vec![w, int64("shape", &[2, 3])],
				input: vec![value("x", &[3, 4]), value("v", &[3])],
				output: vec![value("z", &[2, 3]), value("u", &[4])],
				..Default::default()
			}),
			..Default::default()
		};
		let back = ModelProto::decode(Bytes::from(m.encode_to_vec())).unwrap();
		assert_eq!(back, m);
		let done = patch_model(&mut m).unwrap();
		assert_eq!((done.matmuls, done.reshapes, done.unknown.len()), (2, 1, 0));
		let g = m.graph.unwrap();
		let ops: Vec<&str> = g.node.iter().map(|n| n.op_type.as_deref().unwrap()).collect();
		assert_eq!(ops, ["MatMul", "Squeeze", "Unsqueeze", "MatMul", "Squeeze", "Reshape"]);
		assert_eq!(g.node[0].input, ["x", "w_patch1d_col"]);
		assert_eq!(g.node[1].input, ["y_patch1d_2d", "patch1d_axes_m1"]);
		assert_eq!(g.node[2].input, ["v", "patch1d_axes_0"]);
		assert_eq!(g.node[4].input, ["u_patch1d_2d", "patch1d_axes_m2"]);
		assert_eq!(g.node[5].attribute[0].i, Some(0));
		let names: Vec<&str> = g.initializer.iter().map(|t| t.name.as_deref().unwrap()).collect();
		// the old 1-D weight isn't read any more
		assert_eq!(names, ["shape", "w_patch1d_col", "patch1d_axes_m1", "patch1d_axes_0", "patch1d_axes_m2"]);
		assert_eq!(g.initializer[1].dims, [4, 1]);
	}

	#[test]
	fn varying_sizes_are_named() {
		let mut v = value("pixels", &[1, 3, 64, 48]);
		if let Some(type_proto::Value::TensorType(t)) = v.r#type.as_mut().and_then(|t| t.value.as_mut()) {
			let dim = &mut t.shape.as_mut().unwrap().dim;
			dim[0].value = Some(Dim::DimParam("batch".into()));
			dim[2].value = Some(Dim::DimParam("height".into()));
		}
		let m = ModelProto { graph: Some(GraphProto { input: vec![v], ..Default::default() }), ..Default::default() };
		let got = named_sizes(&m, "pixels:1,3,512,48").unwrap();
		assert_eq!(got, [("batch".to_string(), 1), ("height".to_string(), 512)]);
		// a fixed size that isn't the one asked for, the wrong count, or no such input
		assert!(named_sizes(&m, "pixels:1,3,512,50").is_err());
		assert!(named_sizes(&m, "pixels:1,3").is_err() && named_sizes(&m, "other:1").is_err());
	}
}
