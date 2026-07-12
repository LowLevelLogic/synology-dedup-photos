//! Semantic (embedding-based) image similarity using a CLIP vision model.
//!
//! Unlike the perceptual dHash — which encodes *where* light/dark gradients sit
//! and therefore breaks as soon as the camera moves — CLIP embeddings encode
//! *what is in the picture*, so retakes of the same scene from a slightly
//! different angle land close together in embedding space.

use std::path::PathBuf;
use std::sync::Mutex;

use ort::session::Session;
use ort::value::Tensor;

const MODEL_URL: &str =
    "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx";
const MODEL_FILE: &str = "clip-vit-b32-vision-q8.onnx";

fn model_path() -> PathBuf {
    crate::dirs().join("models").join(MODEL_FILE)
}

/// Download the CLIP vision model on first use (~88 MB, cached forever).
fn ensure_model() -> Result<PathBuf, String> {
    let path = model_path();
    if path.exists() {
        return Ok(path);
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    println!("First run of --semantic: downloading CLIP vision model (~88 MB, one-time)...");
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| e.to_string())?;
    let mut resp = client
        .get(MODEL_URL)
        .send()
        .map_err(|e| format!("Model download failed: {}", e))?
        .error_for_status()
        .map_err(|e| format!("Model download failed: {}", e))?;
    let tmp = path.with_extension("part");
    let mut file = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
    std::io::copy(&mut resp, &mut file).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
    println!("Model saved to {}", path.display());
    Ok(path)
}

pub struct Embedder {
    // ONNX Runtime parallelises internally; a single session behind a mutex
    // keeps the API simple while image decode/preprocess runs on rayon threads.
    session: Mutex<Session>,
}

impl Embedder {
    pub fn new() -> Result<Self, String> {
        let path = ensure_model()?;
        let session = Session::builder()
            .map_err(|e| e.to_string())?
            .commit_from_file(&path)
            .map_err(|e| e.to_string())?;
        Ok(Embedder {
            session: Mutex::new(session),
        })
    }

    /// Standard CLIP preprocessing: shortest side to 224, center crop 224x224,
    /// normalise with CLIP mean/std, NCHW layout.
    fn preprocess(bytes: &[u8]) -> Option<Vec<f32>> {
        const SIZE: u32 = 224;
        const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
        const STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];

        let img = image::load_from_memory(bytes).ok()?;
        let (w, h) = (img.width(), img.height());
        if w == 0 || h == 0 {
            return None;
        }
        let scale = SIZE as f32 / w.min(h) as f32;
        let nw = ((w as f32 * scale).round() as u32).max(SIZE);
        let nh = ((h as f32 * scale).round() as u32).max(SIZE);
        let resized = img
            .resize_exact(nw, nh, image::imageops::FilterType::CatmullRom)
            .to_rgb8();
        let x0 = (nw - SIZE) / 2;
        let y0 = (nh - SIZE) / 2;

        let n = (SIZE * SIZE) as usize;
        let mut out = vec![0f32; 3 * n];
        for y in 0..SIZE {
            for x in 0..SIZE {
                let p = resized.get_pixel(x0 + x, y0 + y);
                let i = (y * SIZE + x) as usize;
                for c in 0..3 {
                    out[c * n + i] = (p[c] as f32 / 255.0 - MEAN[c]) / STD[c];
                }
            }
        }
        Some(out)
    }

    /// Embed an image, returning an L2-normalised vector.
    pub fn embed(&self, bytes: &[u8]) -> Option<Vec<f32>> {
        let input = Self::preprocess(bytes)?;
        let tensor = Tensor::from_array(([1usize, 3, 224, 224], input)).ok()?;
        let mut session = self.session.lock().ok()?;
        let outputs = session.run(ort::inputs!["pixel_values" => tensor]).ok()?;
        let mut v: Vec<f32> = if let Some(value) = outputs.get("image_embeds") {
            value.try_extract_tensor::<f32>().ok()?.1.to_vec()
        } else {
            let (_, value) = outputs.iter().next()?;
            value.try_extract_tensor::<f32>().ok()?.1.to_vec()
        };
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return None;
        }
        for x in v.iter_mut() {
            *x /= norm;
        }
        Some(v)
    }
}

// ── Compact embedding storage ─────────────────────────────────────────────────
// Embeddings are L2-normalised then quantised to i8 and base64-encoded so the
// cache stays small (~700 bytes per photo instead of ~5 KB of JSON floats).

pub fn quantize(v: &[f32]) -> String {
    use base64::{engine::general_purpose, Engine as _};
    let bytes: Vec<u8> = v
        .iter()
        .map(|x| (x * 127.0).round().clamp(-127.0, 127.0) as i8 as u8)
        .collect();
    general_purpose::STANDARD.encode(bytes)
}

pub fn dequantize(s: &str) -> Option<Vec<i8>> {
    use base64::{engine::general_purpose, Engine as _};
    let bytes = general_purpose::STANDARD.decode(s).ok()?;
    Some(bytes.into_iter().map(|b| b as i8).collect())
}

/// Cosine similarity between two quantised embeddings.
pub fn cosine_q(a: &[i8], b: &[i8]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0i64, 0i64, 0i64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as i64 * y as i64;
        na += x as i64 * x as i64;
        nb += y as i64 * y as i64;
    }
    if na == 0 || nb == 0 {
        return 0.0;
    }
    dot as f32 / ((na as f32).sqrt() * (nb as f32).sqrt())
}
