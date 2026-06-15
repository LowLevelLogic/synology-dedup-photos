mod nas;

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::UNIX_EPOCH;
use rayon::prelude::*;
use walkdir::WalkDir;

// ── Constants ─────────────────────────────────────────────────────────────────

pub const IMAGE_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "heic", "heif", "gif", "bmp",
    "tiff", "tif", "webp", "raw", "cr2", "cr3", "nef",
    "arw", "dng", "orf", "rw2", "pef",
];

const BROWSER_DISPLAYABLE: &[&str] = &["jpg", "jpeg", "png", "gif", "bmp", "webp"];

// ── File entry (shared by both modes) ─────────────────────────────────────────

#[derive(Clone)]
pub enum FileSource {
    Local(PathBuf),
    Nas(String), // absolute NAS path, e.g. /volume1/photos/img.jpg
}

#[derive(Clone)]
pub struct FileEntry {
    pub display_path: String,
    pub size: u64,
    pub mtime: u64, // Unix seconds
    pub ext: String, // lowercase, no dot
    pub source: FileSource,
}

// ── Local helpers ─────────────────────────────────────────────────────────────

fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else {
        format!("{:.2} KB", bytes as f64 / 1_024.0)
    }
}

fn hash_local_file(path: &Path) -> io::Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn collect_local_files(root: &str, all_files: bool) -> Vec<FileEntry> {
    let mut files = Vec::new();

    for entry in WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path().to_owned();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();

        if !all_files && !IMAGE_EXTENSIONS.contains(&ext.as_str()) {
            continue;
        }

        if let Ok(meta) = fs::metadata(&path) {
            let size = meta.len();
            if size == 0 {
                continue;
            }
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);

            files.push(FileEntry {
                display_path: path.display().to_string(),
                size,
                mtime,
                ext,
                source: FileSource::Local(path),
            });
        }
    }

    files
}

// ── Perceptual Hashing (dHash) & Similarity ───────────────────────────────────

fn compute_dhash_from_bytes(bytes: &[u8]) -> Option<u64> {
    let img = image::load_from_memory(bytes).ok()?;
    let gray = img.resize_exact(9, 8, image::imageops::FilterType::Nearest).to_luma8();
    let mut hash = 0u64;
    for y in 0..8 {
        for x in 0..8 {
            let left = gray.get_pixel(x, y)[0];
            let right = gray.get_pixel(x + 1, y)[0];
            hash <<= 1;
            if left > right {
                hash |= 1;
            }
        }
    }
    Some(hash)
}

fn compute_dhash_local(path: &Path) -> Option<u64> {
    let bytes = std::fs::read(path).ok()?;
    compute_dhash_from_bytes(&bytes)
}

fn compute_dhash_nas(session: &nas::NasClient, nas_path: &str) -> Option<u64> {
    let (_, bytes) = session.thumbnail_bytes(nas_path)?;
    compute_dhash_from_bytes(&bytes)
}

fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Default similarity threshold: max Hamming distance between two dHash values
/// to consider them "similar". 10 out of 64 bits ≈ 15% difference.
const DEFAULT_THRESHOLD: u32 = 10;

/// Union-Find data structure for clustering similar images.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb { return; }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

fn find_similar(
    files: Vec<FileEntry>,
    hash_fn: impl Fn(&FileEntry) -> Option<u64> + Sync,
    threshold: u32,
) -> (Vec<Vec<FileEntry>>, usize) {
    let n = files.len();

    // ── Load cache ──────────────────────────────────────────────────────────
    let mut cache = HashCache::load();
    let mut hashes: Vec<Option<u64>> = vec![None; n];
    let mut to_hash: Vec<usize> = Vec::new();
    let mut cached_count = 0usize;

    for (i, f) in files.iter().enumerate() {
        if let Some(h) = cache.get(f) {
            hashes[i] = Some(h);
            cached_count += 1;
        } else {
            to_hash.push(i);
        }
    }

    if cached_count > 0 {
        println!("  ⚡ {} files loaded from cache", cached_count);
    }

    // ── Parallel hash uncached files ─────────────────────────────────────────
    let mut errors = 0usize;
    if !to_hash.is_empty() {
        println!("  Hashing {} new files in parallel...", to_hash.len());
        let counter = AtomicUsize::new(0);
        let total = to_hash.len();

        let results: Vec<(usize, Option<u64>)> = to_hash
            .par_iter()
            .map(|&i| {
                let h = hash_fn(&files[i]);
                let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 50 == 0 || done == total {
                    eprint!("\r  Hashed {}/{} files", done, total);
                }
                (i, h)
            })
            .collect();
        eprintln!();

        for (i, h) in results {
            if h.is_none() { errors += 1; }
            hashes[i] = h;
            if let Some(hash_val) = h {
                cache.insert(&files[i], hash_val);
            }
        }

        cache.save();
        println!("  Cache saved ({} total entries)", cache.len());
    }

    // ── Compare all pairs and union similar ones ────────────────────────────
    println!("Comparing all pairs (threshold = {} bits)...", threshold);
    let mut uf = UnionFind::new(n);
    for i in 0..n {
        let h1 = match hashes[i] {
            Some(h) => h,
            None => continue,
        };
        for j in (i + 1)..n {
            let h2 = match hashes[j] {
                Some(h) => h,
                None => continue,
            };
            if hamming_distance(h1, h2) <= threshold {
                uf.union(i, j);
            }
        }
    }

    // ── Collect groups from union-find ───────────────────────────────────────
    let mut group_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        if hashes[i].is_none() { continue; }
        let root = uf.find(i);
        group_map.entry(root).or_default().push(i);
    }

    let groups: Vec<Vec<FileEntry>> = group_map
        .into_values()
        .filter(|indices| indices.len() > 1)
        .map(|indices| indices.into_iter().map(|i| files[i].clone()).collect())
        .collect();

    println!("Found {} groups of similar images", groups.len());
    (groups, errors)
}

// ── Hash Cache ────────────────────────────────────────────────────────────────

struct HashCache {
    path: PathBuf,
    entries: HashMap<String, u64>,
}

impl HashCache {
    fn load() -> Self {
        let cache_dir = dirs();
        let path = cache_dir.join("hash_cache.json");
        let entries = if path.exists() {
            fs::read_to_string(&path)
                .ok()
                .and_then(|data| serde_json::from_str(&data).ok())
                .unwrap_or_default()
        } else {
            HashMap::new()
        };
        HashCache { path, entries }
    }

    fn cache_key(entry: &FileEntry) -> String {
        format!("{}:{}:{}", entry.display_path, entry.size, entry.mtime)
    }

    fn get(&self, entry: &FileEntry) -> Option<u64> {
        self.entries.get(&Self::cache_key(entry)).copied()
    }

    fn insert(&mut self, entry: &FileEntry, hash: u64) {
        self.entries.insert(Self::cache_key(entry), hash);
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn save(&self) {
        if let Some(parent) = self.path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(data) = serde_json::to_string(&self.entries) {
            let _ = fs::write(&self.path, data);
        }
    }
}

fn dirs() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache").join("dedupPictures")
}

// ── Core dedup pipeline ───────────────────────────────────────────────────────

/// Group files by size, hash same-size candidates, return groups of true duplicates.
/// `hash_fn` returns `Some(hex_hash)` or `None` on error.
fn find_duplicates(
    files: Vec<FileEntry>,
    mut hash_fn: impl FnMut(&FileEntry) -> Option<String>,
) -> (Vec<Vec<FileEntry>>, usize) {
    let mut size_groups: HashMap<u64, Vec<FileEntry>> = HashMap::new();
    for f in files {
        size_groups.entry(f.size).or_default().push(f);
    }

    let candidates: Vec<Vec<FileEntry>> = size_groups
        .into_values()
        .filter(|v| v.len() > 1)
        .collect();

    let candidate_count: usize = candidates.iter().map(|v| v.len()).sum();
    println!(
        "Hashing {} files across {} same-size groups...",
        candidate_count,
        candidates.len()
    );

    let mut hash_groups: HashMap<String, Vec<FileEntry>> = HashMap::new();
    let mut errors = 0usize;

    for group in candidates {
        for entry in group {
            match hash_fn(&entry) {
                Some(hash) => hash_groups.entry(hash).or_default().push(entry),
                None => errors += 1,
            }
        }
    }

    let groups = hash_groups
        .into_values()
        .filter(|v| v.len() > 1)
        .collect();

    (groups, errors)
}

fn sort_dup_groups(groups: &mut Vec<Vec<FileEntry>>, strategy: &str) {
    for group in groups.iter_mut() {
        match strategy {
            "newest" => group.sort_by(|a, b| b.mtime.cmp(&a.mtime)),
            "oldest" => group.sort_by(|a, b| a.mtime.cmp(&b.mtime)),
            // Default: keep the largest file first (original vs compressed copies)
            _ => group.sort_by(|a, b| b.size.cmp(&a.size)),
        }
    }
    groups.sort_by(|a, b| a[0].display_path.cmp(&b[0].display_path));
}

fn compute_totals(groups: &[Vec<FileEntry>]) -> (usize, u64) {
    let mut total_del = 0usize;
    let mut total_bytes = 0u64;
    for group in groups {
        for entry in group.iter().skip(1) {
            total_del += 1;
            total_bytes += entry.size;
        }
    }
    (total_del, total_bytes)
}

// ── Terminal report ───────────────────────────────────────────────────────────

fn print_report(groups: &[Vec<FileEntry>], total_del: usize, total_bytes: u64, delete: bool) {
    println!();
    for (i, group) in groups.iter().enumerate() {
        println!("Group {} ({} copies):", i + 1, group.len());
        for (j, entry) in group.iter().enumerate() {
            if j == 0 {
                println!("  [KEEP] {}", entry.display_path);
            } else {
                println!(
                    "  [DEL]  {} ({})",
                    entry.display_path,
                    format_size(entry.size)
                );
            }
        }
    }

    println!();
    println!("--- Summary ---");
    println!("Duplicate groups : {}", groups.len());
    println!("Files to remove  : {}", total_del);
    println!("Space to recover : {}", format_size(total_bytes));

    if !delete {
        println!();
        println!("Dry-run complete. Re-run with --delete to remove the [DEL] files.");
    }
}

// ── HTML preview & Interactive UI ─────────────────────────────────────────────

fn generate_html_preview(
    groups: &[Vec<FileEntry>],
    nas: Option<&nas::NasClient>,
) -> String {
    let mut html = String::new();

    html.push_str(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Duplicate Image Review</title>
<style>
  :root { --bg: #0f172a; --card: #1e293b; --accent: #3b82f6; --text: #f8fafc; --keep: #10b981; --del: #ef4444; }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', system-ui, sans-serif; }
  body { background: var(--bg); color: var(--text); padding-bottom: 100px; }
  header { background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(12px); padding: 2rem; position: sticky; top: 0; z-index: 100; border-bottom: 1px solid rgba(255,255,255,0.1); display: flex; justify-content: space-between; align-items: center; }
  h1 { font-size: 1.8rem; font-weight: 700; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .summary { color: #94a3b8; font-size: 0.95rem; margin-top: 0.5rem; }
  .container { max-width: 1400px; margin: 2rem auto; padding: 0 2rem; }
  .group { background: var(--card); border-radius: 16px; padding: 1.5rem; margin-bottom: 2rem; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); border: 1px solid rgba(255,255,255,0.05); }
  .group-title { font-size: 1.1rem; font-weight: 600; color: #cbd5e1; margin-bottom: 1rem; display: flex; justify-content: space-between;}
  .images { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1.5rem; }
  .card { border-radius: 12px; overflow: hidden; background: #0f172a; cursor: pointer; transition: all 0.2s ease; position: relative; border: 2px solid transparent; }
  .card:hover { transform: translateY(-4px); box-shadow: 0 20px 25px -5px rgba(0,0,0,0.3); }
  .card.keep { border-color: var(--keep); box-shadow: 0 0 0 2px rgba(16,185,129,0.2); }
  .card.del { border-color: var(--del); opacity: 0.6; filter: grayscale(50%); }
  .card.del:hover { opacity: 1; filter: none; }
  .card img { width: 100%; height: 220px; object-fit: cover; display: block; border-bottom: 1px solid rgba(255,255,255,0.05); }
  .badge { position: absolute; top: 12px; right: 12px; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em; backdrop-filter: blur(4px); box-shadow: 0 2px 4px rgba(0,0,0,0.2); transition: all 0.2s; }
  .keep .badge { background: rgba(16,185,129,0.9); color: white; }
  .del .badge { background: rgba(239,68,68,0.9); color: white; }
  .info { padding: 1rem; }
  .path { color: #cbd5e1; font-size: 0.85rem; word-break: break-all; margin-bottom: 0.5rem; line-height: 1.4; }
  .meta { color: #64748b; font-size: 0.8rem; display: flex; justify-content: space-between; }
  .submit-bar { position: fixed; bottom: 0; left: 0; right: 0; background: rgba(30, 41, 59, 0.9); backdrop-filter: blur(16px); padding: 1rem 2rem; border-top: 1px solid rgba(255,255,255,0.1); display: flex; justify-content: space-between; align-items: center; z-index: 1000; transform: translateY(100%); transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1); }
  .submit-bar.visible { transform: translateY(0); }
  .stats { font-size: 1rem; color: #e2e8f0; }
  .stats span { font-weight: 700; color: #ef4444; }
  .btn { background: linear-gradient(135deg, #ef4444, #b91c1c); color: white; border: none; padding: 0.75rem 2rem; border-radius: 8px; font-weight: 600; font-size: 1rem; cursor: pointer; transition: all 0.2s; box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.4); }
  .btn:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.5); }
  .no-preview { width: 100%; height: 220px; display: flex; align-items: center; justify-content: center; color: #555; font-size: 0.85em; background: #1a1a1a; }
</style>
</head>
<body>
<header>
  <div>
    <h1>Duplicate Image Review</h1>
    <div class="summary">Found <b>{GROUPS}</b> groups of identical or similar photos</div>
  </div>
</header>
<div class="container">
"#);

    html = html.replace("{GROUPS}", &groups.len().to_string());

    if nas.is_some() {
        print!("Fetching NAS thumbnails");
        io::stdout().flush().ok();
    }

    for (i, group) in groups.iter().enumerate() {
        html.push_str(&format!(
            r#"<div class="group">
<div class="group-title">Group <span>{}</span> &mdash; {} copies</div>
<div class="images">
"#,
            i + 1,
            group.len()
        ));

        for (j, entry) in group.iter().enumerate() {
            let card_class = if j == 0 { "keep" } else { "del" };
            let badge = if j == 0 { "KEEP" } else { "DELETE" };

            let media = match &entry.source {
                FileSource::Local(path)
                    if crate::BROWSER_DISPLAYABLE.contains(&entry.ext.as_str()) =>
                {
                    let abs = path.canonicalize().unwrap_or_else(|_| path.clone());
                    let url = abs.display().to_string().replace(' ', "%20");
                    format!(r#"<img src="file://{}" alt="" loading="lazy">"#, url)
                }
                FileSource::Nas(nas_path) => {
                    if nas.is_some() {
                        print!(".");
                        io::stdout().flush().ok();
                    }
                    match nas.and_then(|s| s.thumbnail_data_uri(nas_path)) {
                        Some(uri) => format!(r#"<img src="{}" alt="">"#, uri),
                        None => format!(
                            r#"<div class="no-preview">{} — no preview</div>"#,
                            entry.ext.to_uppercase()
                        ),
                    }
                }
                _ => format!(
                    r#"<div class="no-preview">{} — no browser preview</div>"#,
                    entry.ext.to_uppercase()
                ),
            };

            let path_for_js = match &entry.source {
                FileSource::Local(path) => path.display().to_string(),
                FileSource::Nas(p) => p.clone(),
            };

            html.push_str(&format!(
                r#"  <div class="card {}" data-path="{}" data-size="{}">{}
    <div class="badge">{}</div>
    <div class="info">
      <div class="path">{}</div>
      <div class="meta">{} &nbsp;&middot;&nbsp; <span class="ts" data-ts="{}"></span></div>
    </div>
  </div>
"#,
                card_class,
                path_for_js,
                entry.size,
                media,
                badge,
                entry.display_path,
                format_size(entry.size),
                entry.mtime
            ));
        }

        html.push_str("</div></div>\n");
    }

    if nas.is_some() {
        println!(" done");
    }

    html.push_str(
        r#"</div>
<div class="submit-bar" id="submitBar">
  <div class="stats">Ready to delete <span id="delCount">0</span> files &nbsp;&middot;&nbsp; Free up <span id="delSize">0 KB</span></div>
  <button class="btn" id="submitBtn">Delete Selected Files</button>
</div>

<script>
  function formatSize(bytes) {
    if (bytes >= 1073741824) return (bytes / 1073741824).toFixed(2) + ' GB';
    if (bytes >= 1048576) return (bytes / 1048576).toFixed(2) + ' MB';
    return (bytes / 1024).toFixed(2) + ' KB';
  }

  function updateStats() {
    const dels = document.querySelectorAll('.card.del');
    let totalBytes = 0;
    dels.forEach(c => totalBytes += parseInt(c.dataset.size || '0'));
    document.getElementById('delCount').textContent = dels.length;
    document.getElementById('delSize').textContent = formatSize(totalBytes);
    const bar = document.getElementById('submitBar');
    if (dels.length > 0) { bar.classList.add('visible'); } else { bar.classList.remove('visible'); }
  }

  function autosave() {
    const toDelete = Array.from(document.querySelectorAll('.card.del')).map(c => c.dataset.path);
    fetch('/autosave', { method: 'POST', body: JSON.stringify(toDelete) }).catch(() => {});
  }

  document.querySelectorAll('.card').forEach(card => {
    card.addEventListener('click', () => {
      if (card.classList.contains('keep')) {
        card.classList.remove('keep');
        card.classList.add('del');
        card.querySelector('.badge').textContent = 'DELETE';
      } else {
        card.classList.remove('del');
        card.classList.add('keep');
        card.querySelector('.badge').textContent = 'KEEP';
      }
      updateStats();
      autosave();
    });
  });

  updateStats();

  // Restore saved draft selections on page load
  fetch('/draft').then(r => r.json()).then(paths => {
    if (!paths || paths.length === 0) return;
    const deleteSet = new Set(paths);
    document.querySelectorAll('.card').forEach(card => {
      if (deleteSet.has(card.dataset.path)) {
        card.classList.remove('keep');
        card.classList.add('del');
        card.querySelector('.badge').textContent = 'DELETE';
      } else {
        card.classList.remove('del');
        card.classList.add('keep');
        card.querySelector('.badge').textContent = 'KEEP';
      }
    });
    updateStats();
  }).catch(() => {});

  document.getElementById('submitBtn').addEventListener('click', async () => {
    const toDelete = Array.from(document.querySelectorAll('.card.del')).map(c => c.dataset.path);
    if (!confirm(`Are you absolutely sure you want to permanently delete ${toDelete.length} files? This cannot be undone.`)) return;
    
    document.getElementById('submitBtn').textContent = 'Deleting...';
    document.getElementById('submitBtn').disabled = true;

    try {
      const res = await fetch('/delete', { method: 'POST', body: JSON.stringify(toDelete) });
      if (res.ok) {
        document.body.innerHTML = '<div style="display:flex;height:100vh;align-items:center;justify-content:center;flex-direction:column;background:#0f172a;color:#f8fafc;font-family:system-ui"><h1>🎉 Deletion Complete!</h1><p style="color:#94a3b8;margin-top:1rem">You can close this window and check your terminal.</p></div>';
      } else {
        alert('Error deleting files.');
        document.getElementById('submitBtn').textContent = 'Try Again';
        document.getElementById('submitBtn').disabled = false;
      }
    } catch (e) {
      alert('Network error. Don\'t worry — your selections are saved. Use --resume to continue.');
    }
  });

  document.querySelectorAll('.ts').forEach(el => {
    const d = new Date(parseInt(el.dataset.ts) * 1000);
    el.textContent = 'modified ' + d.toLocaleDateString(undefined, {year:'numeric',month:'short',day:'numeric'});
  });
</script>
</body></html>
"#);

    html
}

fn start_web_server(html: String) -> io::Result<Vec<String>> {
    let cache_dir = dirs();
    let _ = fs::create_dir_all(&cache_dir);

    // Save session HTML for --resume
    let _ = fs::write(cache_dir.join("last_session.html"), &html);

    let server = tiny_http::Server::http("127.0.0.1:8080").map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    println!("\n🌐 Interactive preview available at: http://127.0.0.1:8080");
    println!("Your selections are auto-saved. Use --resume to continue if this process stops.");
    open_in_browser("http://127.0.0.1:8080");

    let json_header = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
    let html_header = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap();

    for mut request in server.incoming_requests() {
        match (request.method().as_str(), request.url()) {
            ("GET", "/") => {
                let response = tiny_http::Response::from_string(html.clone())
                    .with_header(html_header.clone());
                let _ = request.respond(response);
            }
            ("GET", "/draft") => {
                let draft_path = cache_dir.join("draft_selection.json");
                let draft = fs::read_to_string(&draft_path).unwrap_or_else(|_| "[]".into());
                let response = tiny_http::Response::from_string(draft)
                    .with_header(json_header.clone());
                let _ = request.respond(response);
            }
            ("POST", "/autosave") => {
                let mut content = String::new();
                let _ = request.as_reader().read_to_string(&mut content);
                let _ = fs::write(cache_dir.join("draft_selection.json"), &content);
                let _ = request.respond(tiny_http::Response::from_string("OK"));
            }
            ("POST", "/delete") => {
                let mut content = String::new();
                request.as_reader().read_to_string(&mut content)?;

                let paths: Vec<String> = serde_json::from_str(&content)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

                let _ = request.respond(tiny_http::Response::from_string("Success"));

                // Clean up session files after successful delete
                let _ = fs::remove_file(cache_dir.join("draft_selection.json"));
                let _ = fs::remove_file(cache_dir.join("last_session.html"));

                return Ok(paths);
            }
            _ => {
                let _ = request.respond(tiny_http::Response::from_string("Not Found").with_status_code(404));
            }
        }
    }
    Ok(vec![])
}

fn open_in_browser(url: &str) {
    #[cfg(target_os = "macos")]
    { let _ = std::process::Command::new("open").arg(url).spawn(); }
    #[cfg(target_os = "linux")]
    { let _ = std::process::Command::new("xdg-open").arg(url).spawn(); }
    #[cfg(target_os = "windows")]
    { let _ = std::process::Command::new("cmd").args(["/c", "start", "", url]).spawn(); }
}

// ── Usage ─────────────────────────────────────────────────────────────────────

fn print_usage(prog: &str) {
    eprintln!("Usage:");
    eprintln!("  {} <path> [OPTIONS]                            (local mode)", prog);
    eprintln!(
        "  {} <nas-path> --nas-host <HOST> --nas-user <USER> [OPTIONS]  (NAS mode)",
        prog
    );
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  {} /Volumes/photos --preview", prog);
    eprintln!(
        "  {} /volume1/photos --nas-host 192.168.1.100 --nas-user admin --preview",
        prog
    );
    eprintln!();
    eprintln!("Common options:");
    eprintln!("  --list-shares        List available root folder paths on the NAS, then exit");
    eprintln!("  --similar            Find visually similar pictures (using perceptual hashing)");
    eprintln!("  --threshold <N>      Hamming distance threshold for --similar (default: 10, max: 64)");
    eprintln!("  --delete             Delete duplicates (default: dry-run)");
    eprintln!("  --keep newest        Keep newest file per group (default: largest file)");
    eprintln!("  --keep oldest        Keep oldest file per group");
    eprintln!("  --all-files          Scan all file types, not just images");
    eprintln!("  --preview            Open an interactive HTML report in your browser");
    eprintln!("  --resume             Resume a previous --preview session (selections are auto-saved)");
    eprintln!("  --clear-cache        Force re-hash all files (clears the perceptual hash cache)");
    eprintln!();
    eprintln!("NAS options (Synology DSM with Secure SignIn / TOTP):");
    eprintln!("  --nas-host <HOST>    NAS address — e.g. 192.168.1.100  or  nas.local:5000");
    eprintln!("                       Defaults to HTTPS on port 5001; port 5000 → HTTP");
    eprintln!("  --nas-user <USER>    DSM username (prompted if omitted)");
    eprintln!("  --nas-otp  <CODE>    6-digit OTP from Synology Secure SignIn app");
    eprintln!("                       (prompted interactively if omitted)");
    eprintln!("  Password is always prompted — never passed as a flag.");
    eprintln!();
    eprintln!("By default runs in dry-run mode. Review the output, then re-run with --delete.");
    eprintln!("Cache location: ~/.cache/dedupPictures/");
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        print_usage(&args[0]);
        std::process::exit(0);
    }

    let path = args[1].clone();
    let delete = args.contains(&"--delete".to_string());
    let all_files = args.contains(&"--all-files".to_string());
    let similar = args.contains(&"--similar".to_string());
    let preview = args.contains(&"--preview".to_string());
    let list_shares = args.contains(&"--list-shares".to_string());
    let resume = args.contains(&"--resume".to_string());
    let clear_cache = args.contains(&"--clear-cache".to_string());

    if clear_cache {
        let cache_file = dirs().join("hash_cache.json");
        if cache_file.exists() {
            let _ = fs::remove_file(&cache_file);
            println!("Cache cleared.");
        }
    }

    let keep_strategy = args
        .windows(2)
        .find(|w| w[0] == "--keep")
        .map(|w| w[1].as_str())
        .unwrap_or("largest");

    if !matches!(keep_strategy, "largest" | "newest" | "oldest") {
        eprintln!("Unknown --keep value '{}'. Use: largest, newest, oldest", keep_strategy);
        std::process::exit(1);
    }

    let threshold: u32 = args
        .windows(2)
        .find(|w| w[0] == "--threshold")
        .map(|w| w[1].parse::<u32>().unwrap_or(DEFAULT_THRESHOLD))
        .unwrap_or(DEFAULT_THRESHOLD);

    let nas_host = args.windows(2).find(|w| w[0] == "--nas-host").map(|w| w[1].clone());
    let nas_user = args.windows(2).find(|w| w[0] == "--nas-user").map(|w| w[1].clone());
    let nas_otp  = args.windows(2).find(|w| w[0] == "--nas-otp" ).map(|w| w[1].clone());

    if let Some(host) = nas_host {
        run_nas(&path, &host, nas_user, nas_otp, delete, all_files, similar, preview, keep_strategy, list_shares, threshold, resume);
    } else {
        run_local(&path, delete, all_files, similar, preview, keep_strategy, threshold, resume);
    }
}

// ── Local mode ────────────────────────────────────────────────────────────────

fn run_local(root: &str, delete: bool, all_files: bool, similar: bool, preview: bool, keep_strategy: &str, threshold: u32, resume: bool) {
    // Resume a previous session
    if resume {
        let session_path = dirs().join("last_session.html");
        if !session_path.exists() {
            eprintln!("No saved session found. Run with --preview first.");
            std::process::exit(1);
        }
        let html = fs::read_to_string(&session_path).expect("Failed to read saved session");
        println!("Resuming previous session...");
        if let Ok(to_delete) = start_web_server(html) {
            if to_delete.is_empty() { return; }
            println!("\nDeleting {} files...", to_delete.len());
            let (mut deleted, mut errs) = (0usize, 0usize);
            for path_str in to_delete {
                match fs::remove_file(&path_str) {
                    Ok(_) => { println!("  Deleted: {}", path_str); deleted += 1; }
                    Err(e) => { eprintln!("  Error: {} \u{2014} {}", path_str, e); errs += 1; }
                }
            }
            println!("Done. Deleted {} files ({} errors).", deleted, errs);
        }
        return;
    }

    if !Path::new(root).exists() {
        eprintln!("Path does not exist: {}", root);
        std::process::exit(1);
    }

    println!("Scanning : {} (local)", root);
    println!("Mode     : {}", if delete { "DELETE" } else { "dry-run" });
    println!("Keep     : {}", keep_strategy);
    println!();

    let files = collect_local_files(root, all_files);
    println!("Scanned {} files", files.len());

    let (mut groups, errors) = if similar {
        find_similar(files, |entry| {
            if let FileSource::Local(path) = &entry.source {
                compute_dhash_local(path)
            } else {
                None
            }
        }, threshold)
    } else {
        find_duplicates(files, |entry| {
            if let FileSource::Local(path) = &entry.source {
                match hash_local_file(path) {
                    Ok(h) => Some(h),
                    Err(e) => {
                        eprintln!("  Warning: cannot hash {:?}: {}", path, e);
                        None
                    }
                }
            } else {
                None
            }
        })
    };

    if errors > 0 {
        eprintln!("  ({} files skipped due to errors)", errors);
    }
    if groups.is_empty() {
        println!("\nNo duplicates found.");
        return;
    }

    sort_dup_groups(&mut groups, keep_strategy);
    let (total_del, total_bytes) = compute_totals(&groups);
    print_report(&groups, total_del, total_bytes, delete);

    if preview {
        let html = generate_html_preview(&groups, None);
        if let Ok(to_delete) = start_web_server(html) {
            if to_delete.is_empty() {
                println!("No files selected for deletion.");
                return;
            }
            println!("\nDeleting {} files from UI request...", to_delete.len());
            let (mut deleted, mut errs) = (0usize, 0usize);
            for path_str in to_delete {
                match fs::remove_file(&path_str) {
                    Ok(_) => { println!("  Deleted: {}", path_str); deleted += 1; }
                    Err(e) => { eprintln!("  Error: {} — {}", path_str, e); errs += 1; }
                }
            }
            println!("Done. Deleted {} files ({} errors).", deleted, errs);
        }
    } else if delete {
        println!();
        println!("Deleting...");
        let (mut deleted, mut errs) = (0usize, 0usize);
        for group in &groups {
            for entry in group.iter().skip(1) {
                if let FileSource::Local(path) = &entry.source {
                    match fs::remove_file(path) {
                        Ok(_) => { println!("  Deleted: {}", entry.display_path); deleted += 1; }
                        Err(e) => { eprintln!("  Error: {} — {}", entry.display_path, e); errs += 1; }
                    }
                }
            }
        }
        println!();
        println!("Done. Deleted {} files ({} errors).", deleted, errs);
    }
}

// ── NAS mode ──────────────────────────────────────────────────────────────────

fn run_nas(
    nas_path: &str,
    host: &str,
    nas_user: Option<String>,
    nas_otp_flag: Option<String>,
    delete: bool,
    all_files: bool,
    similar: bool,
    preview: bool,
    keep_strategy: &str,
    list_shares: bool,
    threshold: u32,
    resume: bool,
) {
    let base_url = nas::build_base_url(host);

    // Collect credentials — password is always prompted to keep it out of shell history
    let user = match nas_user {
        Some(u) => u,
        None => {
            eprint!("NAS username: ");
            io::stderr().flush().ok();
            let mut u = String::new();
            io::stdin().read_line(&mut u).expect("Failed to read username");
            u.trim().to_string()
        }
    };

    let password = rpassword::prompt_password("NAS password: ")
        .expect("Failed to read password");

    let otp = match nas_otp_flag {
        Some(o) => o,
        None => {
            eprint!("OTP code (from Synology Secure SignIn app, or Enter to skip): ");
            io::stderr().flush().ok();
            let mut o = String::new();
            io::stdin().read_line(&mut o).expect("Failed to read OTP");
            o.trim().to_string()
        }
    };

    println!();
    println!("Connecting : {}", base_url);

    let session = match nas::NasClient::login(&base_url, &user, &password, &otp) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Authentication failed: {}", e);
            std::process::exit(1);
        }
    };

    println!("Authenticated");
    println!();

    if resume {
        let session_path = dirs().join("last_session.html");
        if !session_path.exists() {
            eprintln!("No saved session found. Run with --preview first.");
            session.logout();
            std::process::exit(1);
        }
        let html = fs::read_to_string(&session_path).expect("Failed to read saved session");
        println!("Resuming previous NAS session...");
        if let Ok(to_delete) = start_web_server(html) {
            if to_delete.is_empty() { 
                session.logout();
                return; 
            }
            println!("\nDeleting {} files from NAS via UI request...", to_delete.len());
            let (mut deleted, mut errs) = (0usize, 0usize);

            let failures = session.delete_files(&to_delete.iter().map(|s| s.as_str()).collect::<Vec<_>>());
            let failed: std::collections::HashSet<&str> = failures.iter().map(|(p, _)| p.as_str()).collect();

            for path in &to_delete {
                if failed.contains(path.as_str()) {
                    errs += 1;
                } else {
                    println!("  Deleted: {}", path);
                    deleted += 1;
                }
            }
            for (p, e) in &failures {
                eprintln!("  Error: {} — {}", p, e);
            }
            println!("Done. Deleted {} files ({} errors).", deleted, errs);
        }
        session.logout();
        return;
    }

    if list_shares {
        println!("Available root folders for user '{}':", user);
        match session.list_shares() {
            Ok(shares) => {
                if shares.is_empty() {
                    println!("  (No shared folders found)");
                }
                for s in shares {
                    println!("  {}", s);
                }
            }
            Err(e) => eprintln!("Failed to list shares: {}", e),
        }
        session.logout();
        return;
    }

    println!("Scanning : {} (NAS)", nas_path);
    println!("Mode     : {}", if delete { "DELETE" } else { "dry-run" });
    println!("Keep     : {}", keep_strategy);
    println!();

    let files = match session.list_recursive(nas_path, all_files) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to list files: {}", e);
            session.logout();
            std::process::exit(1);
        }
    };

    println!("Scanned {} files", files.len());

    let (mut groups, errors) = if similar {
        find_similar(files, |entry| {
            if let FileSource::Nas(p) = &entry.source {
                compute_dhash_nas(&session, p)
            } else {
                None
            }
        }, threshold)
    } else {
        find_duplicates(files, |entry| {
            if let FileSource::Nas(p) = &entry.source {
                match session.hash_file(p) {
                    Ok(h) => Some(h),
                    Err(e) => {
                        eprintln!("  Warning: cannot hash {}: {}", p, e);
                        None
                    }
                }
            } else {
                None
            }
        })
    };

    if errors > 0 {
        eprintln!("  ({} files skipped due to errors)", errors);
    }
    if groups.is_empty() {
        println!("\nNo duplicates found.");
        session.logout();
        return;
    }

    sort_dup_groups(&mut groups, keep_strategy);
    let (total_del, total_bytes) = compute_totals(&groups);
    print_report(&groups, total_del, total_bytes, delete);

    if preview {
        let html = generate_html_preview(&groups, Some(&session));
        if let Ok(to_delete) = start_web_server(html) {
            if to_delete.is_empty() {
                println!("No files selected for deletion.");
                session.logout();
                return;
            }
            println!("\nDeleting {} files from NAS via UI request...", to_delete.len());
            let (mut deleted, mut errs) = (0usize, 0usize);

            let failures = session.delete_files(&to_delete.iter().map(|s| s.as_str()).collect::<Vec<_>>());
            let failed: std::collections::HashSet<&str> = failures.iter().map(|(p, _)| p.as_str()).collect();

            for path in &to_delete {
                if failed.contains(path.as_str()) {
                    errs += 1;
                } else {
                    println!("  Deleted: {}", path);
                    deleted += 1;
                }
            }
            for (p, e) in &failures {
                eprintln!("  Error: {} — {}", p, e);
            }
            println!("Done. Deleted {} files ({} errors).", deleted, errs);
        }
    } else if delete {
        println!();
        println!("Deleting from NAS...");
        let (mut deleted, mut errs) = (0usize, 0usize);

        for group in &groups {
            let to_del: Vec<&str> = group
                .iter()
                .skip(1)
                .filter_map(|e| {
                    if let FileSource::Nas(p) = &e.source { Some(p.as_str()) } else { None }
                })
                .collect();

            if to_del.is_empty() {
                continue;
            }

            let failures = session.delete_files(&to_del);
            let failed: std::collections::HashSet<&str> =
                failures.iter().map(|(p, _)| p.as_str()).collect();

            for path in &to_del {
                if failed.contains(path) {
                    errs += 1;
                } else {
                    println!("  Deleted: {}", path);
                    deleted += 1;
                }
            }
            for (p, e) in &failures {
                eprintln!("  Error: {} — {}", p, e);
            }
        }

        println!();
        println!("Done. Deleted {} files ({} errors).", deleted, errs);
    }

    session.logout();
}
