use base64::{engine::general_purpose, Engine as _};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::time::Duration;

// ── DSM JSON envelope ─────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct DsmResp<T> {
    success: bool,
    data: Option<T>,
    error: Option<DsmErr>,
}

#[derive(Deserialize)]
struct DsmErr {
    code: u32,
}

// ── Per-endpoint response bodies ──────────────────────────────────────────────

#[derive(Deserialize)]
struct AuthData {
    sid: String,
}

#[derive(Deserialize)]
struct ListData {
    files: Vec<NasFileRaw>,
    total: Option<usize>,
}

#[derive(Deserialize)]
struct ListShareData {
    shares: Vec<NasShareRaw>,
}

#[derive(Deserialize)]
struct NasShareRaw {
    path: String,
}

#[derive(Deserialize)]
struct NasFileRaw {
    path: String,
    isdir: bool,
    additional: Option<NasFileAdditional>,
}

#[derive(Deserialize)]
struct NasFileAdditional {
    size: u64,
    time: NasFileTime,
}

#[derive(Deserialize)]
struct NasFileTime {
    mtime: u64,
}



// ── NasClient ─────────────────────────────────────────────────────────────────

pub struct NasClient {
    client: reqwest::blocking::Client,
    base_url: String,
    pub sid: String,
}

fn auth_error(code: u32) -> String {
    match code {
        400 => "Invalid credentials".into(),
        401 => "Guest or disabled account".into(),
        403 => "Permission denied".into(),
        404 => "Account not found".into(),
        406 => "2FA required — add --nas-otp <code>".into(),
        407 => "IP blocked".into(),
        408 => "Invalid OTP code".into(),
        409 => "OTP already used — wait for the next 30-second window".into(),
        410 => "Account locked (too many failed attempts)".into(),
        411 => "Account expired".into(),
        _ => format!("Auth failed (DSM error code {})", code),
    }
}

impl NasClient {
    /// Authenticate against Synology DSM and return a live session.
    /// `otp` may be empty if 2FA is not required.
    pub fn login(base_url: &str, user: &str, pass: &str, otp: &str) -> Result<Self, String> {
        let client = reqwest::blocking::Client::builder()
            // Home NAS setups commonly use self-signed TLS certificates.
            .danger_accept_invalid_certs(true)
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| e.to_string())?;

        let url = format!("{}/webapi/entry.cgi", base_url);

        let mut form: Vec<(&str, &str)> = vec![
            ("api", "SYNO.API.Auth"),
            ("version", "7"),
            ("method", "login"),
            ("account", user),
            ("passwd", pass),
            ("format", "sid"),
            ("session", "FileStation"),
        ];
        if !otp.is_empty() {
            form.push(("otp_code", otp));
        }

        let resp: DsmResp<AuthData> = client
            .post(&url)
            .form(&form)
            .send()
            .map_err(|e| format!("Could not reach NAS: {}", e))?
            .json()
            .map_err(|e| format!("Unexpected response from DSM: {}", e))?;

        if !resp.success {
            let code = resp.error.as_ref().map(|e| e.code).unwrap_or(0);
            return Err(auth_error(code));
        }

        let sid = resp
            .data
            .ok_or("No session data in login response")?
            .sid;

        Ok(NasClient {
            client,
            base_url: base_url.to_string(),
            sid,
        })
    }

    /// List all root shared folders accessible to this user.
    pub fn list_shares(&self) -> Result<Vec<String>, String> {
        let url = format!("{}/webapi/entry.cgi", self.base_url);
        let resp: DsmResp<ListShareData> = self
            .client
            .get(&url)
            .query(&[
                ("api", "SYNO.FileStation.List"),
                ("version", "2"),
                ("method", "list_share"),
                ("_sid", &self.sid),
            ])
            .send()
            .map_err(|e| e.to_string())?
            .json()
            .map_err(|e| e.to_string())?;

        if !resp.success {
            let code = resp.error.as_ref().map(|e| e.code).unwrap_or(0);
            return Err(format!("Cannot list shares (DSM code {})", code));
        }

        let data = resp.data.ok_or_else(|| "Empty response for list_share".to_string())?;
        Ok(data.shares.into_iter().map(|s| s.path).collect())
    }

    /// Recursively list all files under `folder` on the NAS.
    /// If `all_files` is false, only image extensions are returned.
    pub fn list_recursive(
        &self,
        folder: &str,
        all_files: bool,
    ) -> Result<Vec<crate::FileEntry>, String> {
        let mut out = Vec::new();
        self.list_inner(folder, all_files, &mut out)?;
        Ok(out)
    }

    fn list_inner(
        &self,
        folder: &str,
        all_files: bool,
        out: &mut Vec<crate::FileEntry>,
    ) -> Result<(), String> {
        let url = format!("{}/webapi/entry.cgi", self.base_url);
        let mut offset = 0usize;

        loop {
            let resp: DsmResp<ListData> = self
                .client
                .get(&url)
                .query(&[
                    ("api", "SYNO.FileStation.List"),
                    ("version", "2"),
                    ("method", "list"),
                    ("folder_path", folder),
                    ("additional", r#"["size","time"]"#),
                    ("limit", "1000"),
                    ("offset", &offset.to_string()),
                    ("_sid", &self.sid),
                ])
                .send()
                .map_err(|e| e.to_string())?
                .json()
                .map_err(|e| e.to_string())?;

            if !resp.success {
                let code = resp.error.as_ref().map(|e| e.code).unwrap_or(0);
                // 408 = folder not found / no permission
                if code == 408 || code == 1 {
                    eprintln!("Warning: Skipping '{}' (folder not found or permission denied, DSM code {})", folder, code);
                    return Ok(());
                }
                return Err(format!("Cannot list '{}' (DSM code {})", folder, code));
            }

            let data = resp
                .data
                .ok_or_else(|| format!("Empty response for '{}'", folder))?;
            let total = data.total.unwrap_or(usize::MAX);
            let count = data.files.len();

            for f in data.files {
                if f.isdir {
                    self.list_inner(&f.path, all_files, out)?;
                    continue;
                }

                let add = match f.additional {
                    Some(a) => a,
                    None => continue,
                };
                if add.size == 0 {
                    continue;
                }

                let ext = f
                    .path
                    .rsplit('.')
                    .next()
                    .unwrap_or("")
                    .to_lowercase();

                if !all_files && !crate::IMAGE_EXTENSIONS.contains(&ext.as_str()) {
                    continue;
                }

                out.push(crate::FileEntry {
                    display_path: f.path.clone(),
                    size: add.size,
                    mtime: add.time.mtime,
                    ext,
                    source: crate::FileSource::Nas(f.path),
                });
            }

            offset += count;
            if count == 0 || offset >= total {
                break;
            }
        }

        Ok(())
    }

    /// Download a NAS file and return its SHA-256 hex digest.
    /// The file is streamed — never fully loaded into memory.
    pub fn hash_file(&self, nas_path: &str) -> Result<String, String> {
        let url = format!("{}/webapi/entry.cgi", self.base_url);
        let mut resp = self
            .client
            .get(&url)
            .query(&[
                ("api", "SYNO.FileStation.Download"),
                ("version", "2"),
                ("method", "download"),
                ("path", nas_path),
                ("mode", "download"),
                ("_sid", &self.sid),
            ])
            .send()
            .map_err(|e| format!("Download '{}': {}", nas_path, e))?;

        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            let n = resp.read(&mut buf).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }

        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Fetch a thumbnail from the NAS and return raw bytes,
    /// or `None` if the thumbnail is unavailable.
    pub fn thumbnail_bytes(&self, nas_path: &str) -> Option<(String, Vec<u8>)> {
        let url = format!("{}/webapi/entry.cgi", self.base_url);
        let resp = self
            .client
            .get(&url)
            .timeout(Duration::from_secs(10))
            .query(&[
                ("api", "SYNO.FileStation.Thumb"),
                ("version", "2"),
                ("method", "get"),
                ("path", nas_path),
                ("size", "large"),
                ("_sid", &self.sid),
            ])
            .send()
            .ok()?;

        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .to_string();

        if !ct.starts_with("image/") {
            return None;
        }

        let bytes = resp.bytes().ok()?;
        if bytes.is_empty() {
            return None;
        }

        Some((ct, bytes.to_vec()))
    }

    /// Fetch a thumbnail from the NAS and return a base64 data URI,
    /// or `None` if the thumbnail is unavailable.
    pub fn thumbnail_data_uri(&self, nas_path: &str) -> Option<String> {
        self.thumbnail_bytes(nas_path).map(|(ct, bytes)| {
            format!("data:{};base64,{}", ct, general_purpose::STANDARD.encode(&bytes))
        })
    }

    /// Delete files on the NAS via the FileStation Delete API.
    /// Returns a list of `(path, error_message)` for any that failed.
    pub fn delete_files(&self, paths: &[&str]) -> Vec<(String, String)> {
        let mut errors: Vec<(String, String)> = Vec::new();
        let url = format!("{}/webapi/entry.cgi", self.base_url);
        let paths_json = serde_json::to_string(paths).unwrap_or_default();

        let start: DsmResp<serde_json::Value> = match self
            .client
            .post(&url)
            .form(&[
                ("api", "SYNO.FileStation.Delete"),
                ("version", "2"),
                ("method", "start"),
                ("path", &paths_json),
                ("force", "true"),
                ("recursive", "true"),
                ("_sid", &self.sid),
            ])
            .send()
            .and_then(|r| r.json())
        {
            Ok(r) => r,
            Err(e) => {
                let msg = e.to_string();
                for p in paths {
                    errors.push((p.to_string(), msg.clone()));
                }
                return errors;
            }
        };

        if !start.success {
            let msg = format!(
                "Delete start failed (DSM code {})",
                start.error.as_ref().map(|e| e.code).unwrap_or(0)
            );
            for p in paths {
                errors.push((p.to_string(), msg.clone()));
            }
            return errors;
        }

        let taskid = match start.data.and_then(|d| d.get("taskid").and_then(|v| v.as_str()).map(|s| s.to_string())) {
            Some(t) => t,
            None => {
                for p in paths {
                    errors.push((p.to_string(), "No taskid returned".into()));
                }
                return errors;
            }
        };

        // Poll until the task finishes (up to ~60 s)
        for _ in 0..120 {
            std::thread::sleep(Duration::from_millis(500));

            let status: Result<DsmResp<serde_json::Value>, _> = self
                .client
                .get(&url)
                .query(&[
                    ("api", "SYNO.FileStation.Delete"),
                    ("version", "2"),
                    ("method", "status"),
                    ("taskid", &taskid),
                    ("_sid", &self.sid),
                ])
                .send()
                .and_then(|r| r.json());

            if let Ok(s) = status {
                if s.data.and_then(|d| d.get("finished").and_then(|v| v.as_bool())).unwrap_or(false) {
                    return errors; // empty = all succeeded
                }
            }
        }

        for p in paths {
            errors.push((p.to_string(), "Delete task timed out".into()));
        }
        errors
    }

    pub fn logout(&self) {
        let url = format!("{}/webapi/entry.cgi", self.base_url);
        let _ = self
            .client
            .get(&url)
            .query(&[
                ("api", "SYNO.API.Auth"),
                ("version", "1"),
                ("method", "logout"),
                ("session", "FileStation"),
                ("_sid", &self.sid),
            ])
            .send();
    }
}

// ── URL helper ────────────────────────────────────────────────────────────────

/// Normalise a NAS host string into a base URL.
///
/// | Input                        | Output                          |
/// |------------------------------|---------------------------------|
/// | `192.168.1.100`              | `https://192.168.1.100:5001`    |
/// | `192.168.1.100:5000`         | `http://192.168.1.100:5000`     |
/// | `nas.local:5001`             | `https://nas.local:5001`        |
/// | `http://nas.local:5000`      | `http://nas.local:5000`         |
/// | `https://mynas.example.com`  | `https://mynas.example.com`     |
pub fn build_base_url(host: &str) -> String {
    if host.starts_with("http://") || host.starts_with("https://") {
        return host.trim_end_matches('/').to_string();
    }
    // Detect explicit port
    if let Some(port_str) = host.rsplit(':').next() {
        if let Ok(port) = port_str.parse::<u16>() {
            let scheme = if port == 5000 || port == 80 { "http" } else { "https" };
            return format!("{}://{}", scheme, host);
        }
    }
    // Default: HTTPS on Synology's default DSM HTTPS port
    format!("https://{}:5001", host)
}
