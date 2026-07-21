use candle_core::{Device, Tensor};
use serde_json::json;
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"KVFM1\0\0\0";
const HEADER_BYTES: usize = 32;
const OUTPUTS: u32 = 7;

type AnyError = Box<dyn Error + Send + Sync>;
type Result<T> = std::result::Result<T, AnyError>;

const PARAMETERS: [(&str, &[usize]); 10] = [
    ("letter_embedding.weight", &[27, 8]),
    ("boundary_embedding.weight", &[3, 3]),
    ("row_1.weight", &[64, 145]),
    ("row_1.bias", &[64]),
    ("row_2.weight", &[16, 64]),
    ("row_2.bias", &[16]),
    ("head_1.weight", &[64, 69]),
    ("head_1.bias", &[64]),
    ("head_2.weight", &[7, 64]),
    ("head_2.bias", &[7]),
];

fn absolute(value: &str) -> Result<PathBuf> {
    let path = PathBuf::from(value);
    Ok(if path.is_absolute() {
        path
    } else {
        env::current_dir()?.join(path)
    })
}

fn parse_options() -> Result<(PathBuf, PathBuf)> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if !arguments.len().is_multiple_of(2) {
        return Err("expected --input MODEL --output MODEL".into());
    }
    let mut input = None;
    let mut output = None;
    for pair in arguments.chunks_exact(2) {
        match pair[0].as_str() {
            "--input" => input = Some(absolute(&pair[1])?),
            "--output" => output = Some(absolute(&pair[1])?),
            other => return Err(format!("unsupported option {other:?}").into()),
        }
    }
    Ok((
        input.ok_or("--input is required")?,
        output.ok_or("--output is required")?,
    ))
}

fn tensor_values(
    tensors: &HashMap<String, Tensor>,
    name: &str,
    dimensions: &[usize],
) -> Result<Vec<f32>> {
    let tensor = tensors
        .get(name)
        .ok_or_else(|| format!("model lacks {name}"))?;
    if tensor.dims() != dimensions {
        return Err(format!(
            "{name} has dimensions {:?}, expected {dimensions:?}",
            tensor.dims()
        )
        .into());
    }
    Ok(tensor.flatten_all()?.to_vec1::<f32>()?)
}

fn export(input: &Path, output: &Path) -> Result<()> {
    let tensors = candle_core::safetensors::load(input, &Device::Cpu)?;
    let float_count: usize = PARAMETERS
        .iter()
        .map(|(_, dimensions)| dimensions.iter().product::<usize>())
        .sum();
    fs::create_dir_all(output.parent().ok_or("output has no parent")?)?;
    let mut writer = BufWriter::new(File::create(output)?);
    writer.write_all(MAGIC)?;
    writer.write_all(&1u32.to_le_bytes())?;
    writer.write_all(&(float_count as u32).to_le_bytes())?;
    writer.write_all(&OUTPUTS.to_le_bytes())?;
    writer.write_all(&(PARAMETERS.len() as u32).to_le_bytes())?;
    writer.write_all(&[0; HEADER_BYTES - 24])?;
    for (name, dimensions) in PARAMETERS {
        for value in tensor_values(&tensors, name, dimensions)? {
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    writer.flush()?;
    let bytes = fs::metadata(output)?.len();
    println!(
        "{}",
        json!({
            "status": "complete",
            "input": input,
            "output": output,
            "parameters": float_count,
            "bytes": bytes,
        })
    );
    Ok(())
}

fn main() {
    if let Err(error) = parse_options().and_then(|(input, output)| export(&input, &output)) {
        eprintln!("kvadrat-fragment-export: {error}");
        std::process::exit(1);
    }
}
