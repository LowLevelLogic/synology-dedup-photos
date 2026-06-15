# dedupPictures 📸

A blazingly fast, visually-aware photo deduplication tool built in Rust. It finds exact duplicates or visually similar photos (like WhatsApp-compressed copies) both locally on your machine and remotely over the network on a Synology NAS.

It features a beautiful interactive web-based dashboard for reviewing and confirming deletions before they happen.

## Features ✨

* **Perceptual Hashing (dHash):** Don't just find exact byte-for-byte duplicates. `dedupPictures` uses perceptual hashing to find photos that *look* the same, even if one was resized, compressed, or sent over WhatsApp.
* **Synology NAS Native:** Talks directly to the Synology DSM `FileStation` APIs. No need to mount SMB shares or pull terabytes of data over your network. It pulls tiny thumbnails directly from the NAS for blazing fast processing.
* **Interactive Web Dashboard:** Creates a stunning, glassmorphic dark-mode web dashboard on a local server (`http://127.0.0.1:8080`) where you can visually click and toggle which images to Keep or Delete.
* **Session Persistence & Auto-save:** Accidentally closed the terminal or browser while reviewing thousands of photos? All clicks are auto-saved. Pass `--resume` to pick up exactly where you left off.
* **Parallel Processing & Caching:** Uses `rayon` to download and hash photos across all your CPU cores. Hashes are permanently cached to `~/.cache/dedupPictures/`, so second runs are nearly instant!
* **Dry Run by Default:** It will never delete anything unless you explicitly give it permission via the UI or the `--delete` flag.

## Usage 🚀

### Local Mode (Mac / PC)
Point it at a local directory on your machine:
```bash
cargo run --release -- /Users/name/Pictures --preview
```

### NAS Mode (Synology)
Point it at a shared folder path on your Synology NAS:
```bash
cargo run --release -- /home/Photos/iPhone_backup \
  --nas-host 192.168.1.100 \
  --nas-user myusername \
  --similar \
  --preview
```
*(You will be securely prompted for your DSM Password and 2FA OTP code).*

## Flags & Options ⚙️

### Common Flags
| Flag | Description |
|---|---|
| `--similar` | Finds visually similar pictures (using perceptual hashing) instead of exact byte-for-byte duplicates. Highly recommended for photo libraries. |
| `--threshold <N>` | The Hamming distance threshold for `--similar`. Default is `10` (out of 64 bits). Set lower (e.g. `5`) to be strict, set higher (e.g. `15`) to catch heavier compressions. |
| `--preview` | Opens the interactive visual web report in your browser for manual review. |
| `--resume` | Resumes a previous `--preview` session. Picks up your auto-saved KEEP/DELETE selections exactly where you left off. |
| `--keep <strategy>` | Determines which file in a duplicate group is marked to KEEP by default. Options: `largest` (default, good for keeping original high-res over compressed copies), `newest`, `oldest`. |
| `--delete` | Runs strictly in the terminal and automatically deletes all files marked as duplicates. Bypasses the UI. |
| `--clear-cache` | Wipes the perceptual hash cache (`~/.cache/dedupPictures/`) and forces a full re-hash of all files. |
| `--all-files` | Scans all file types instead of filtering for standard image extensions. (Useful with exact byte-for-byte deduplication). |
| `--list-shares` | (NAS Only) Queries the NAS and prints out all available root folder paths that your user has permission to scan. |

### NAS Authentication Flags
| Flag | Description |
|---|---|
| `--nas-host <IP>` | The NAS IP or Hostname (e.g., `10.0.0.38` or `nas.local:5000`). |
| `--nas-user <USER>` | Your Synology DSM username. |
| `--nas-otp <CODE>` | (Optional) Your 6-digit Synology Secure SignIn app code. If omitted, you will be prompted interactively. |

*Note: For security, your password is never passed as a flag. The tool will always prompt you interactively.*
