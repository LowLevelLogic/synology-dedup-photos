# dedupPictures 📸

A blazingly fast, visually-aware photo deduplication tool built in Rust. It finds exact duplicates or visually similar photos (like WhatsApp-compressed copies) both locally on your machine, on generic NAS drives (via network mounts), and via a dedicated ultra-fast API mode for Synology NAS setups.

It features a beautiful interactive web-based dashboard for reviewing and confirming deletions before they happen.

## Features ✨

* **127-bit Dual-Gradient Perceptual Hashing:** Don't just find exact byte-for-byte duplicates. `dedupPictures` uses a custom 127-bit perceptual hash (tracking both horizontal and vertical gradients) to find photos that *look* the same, even if one was resized, compressed, or sent over WhatsApp.
* **Synology NAS Native Architecture:** Most standard photo deduplication tools require you to mount your NAS as a network drive (SMB) and will download the *entire* 50MB+ Raw file for every single photo over your Wi-Fi just to analyze it, choking your network and taking hours. `dedupPictures` talks directly to the Synology DSM `FileStation` APIs. Instead of downloading the original photos, it asks the NAS to generate a tiny 10KB thumbnail and analyzes that instead. This allows it to perceptually scan terabytes of network photos in mere seconds.
* **MFA / 2FA Support:** Fully supports Synology Secure SignIn / Multi-Factor Authentication. If you don't have MFA enabled on your NAS account, simply hit Enter when prompted—it works seamlessly with or without it.
* **Interactive Web Dashboard:** Creates a stunning, glassmorphic dark-mode web dashboard on a local server (`http://127.0.0.1:8080`) where you can visually click and toggle which images to Keep or Delete.
* **Session Persistence & Auto-save:** Accidentally closed the terminal or browser while reviewing thousands of photos? All clicks are auto-saved. Pass `--resume` to pick up exactly where you left off.
* **Parallel Processing & Caching:** Uses `rayon` to download and hash photos across all your CPU cores. Hashes are permanently cached to `~/.cache/dedupPictures/`, so second runs are nearly instant!
* **Dry Run by Default:** It will never delete anything unless you explicitly give it permission via the UI or the `--delete` flag.

## Installation 🛠️

Since `dedupPictures` is built in Rust, you'll need the Rust compiler installed to run it. 

1. **Install Rust:** 
   Open your terminal and run the official Rust installer (rustup):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
   *(Follow the on-screen prompts and restart your terminal afterwards).*

2. **Run the App:** 
   Navigate to the project folder and use `cargo run` (this will automatically compile and launch the app in one step).

## Understanding the Threshold (127-bit dHash)

`dedupPictures` uses a custom **127-bit Dual-Gradient Perceptual Hash (dHash)**. 
Unlike standard cryptographic hashes (SHA-256) which change completely if a single pixel is altered, this perceptual hash tracks the physical transitions of light-to-dark (gradients) across the image layout. 

When you specify a `--threshold` flag (default is `10`), you are setting the maximum **Hamming distance** (the number of differing bits) allowed between two hashes for them to be considered duplicates. Because the algorithm checks if the difference is *less than or equal to* the threshold, it will always catch mathematically exact copies regardless of what you set the threshold to.

- **`--threshold 0`**: Requires exact identical gradients. (Use for finding exact, mathematically identical, uncompressed copies).
- **`--threshold 10` (Recommended & Default)**: Catches everything up to an ~8% variance (0 to 10 differing bits). This is the sweet spot! It finds all of your mathematically exact copies, while also catching WhatsApp compressions, minor resolution drops, and subtle burst-shot variations.
- **`--threshold 20+`**: Very loose. Use with caution, as it may begin to group visually similar but completely distinct photos (like different photos of the exact same landscape).

## Usage 🚀
### Local Mode (Mac / PC)
Point it at a local directory on your machine:
```bash
cargo run --release -- /Users/name/Pictures --preview
```

### Generic NAS Mode (QNAP, TrueNAS, Unraid, etc.)
Because generic network protocols (like WebDAV or SMB) don't have standardized ways to ask the NAS to generate and send tiny thumbnails, building a dedicated network API for them would force the script to download full 10MB+ Raw files over your Wi-Fi, defeating the purpose of a fast network deduplicator. 

Instead, the fastest way to deduplicate a generic NAS is to natively mount it to your computer using SMB or NFS, and run the script in **Local Mode**. The script will seamlessly process it over the network mount.

**How to mount your NAS:**
- **macOS:** Open Finder, press `Cmd + K` (Connect to Server), enter `smb://<NAS_IP>` (e.g., `smb://192.168.1.100`), and enter your NAS credentials. The drive will mount under `/Volumes/`.
- **Windows:** Open File Explorer, right-click "This PC", select "Map network drive...", enter `\\<NAS_IP>\<ShareName>`, and enter your NAS credentials. It will map to a drive letter like `Z:`.

Once mounted, simply run the script against the new local path:
```bash
# macOS Example:
cargo run --release -- /Volumes/MyNAS/Photos --preview

# Windows Example:
cargo run --release -- Z:\Photos --preview
```

### Dedicated NAS Mode (Synology)
If you have a Synology NAS, you're in luck! `dedupPictures` has a dedicated mode that talks directly to the Synology DSM `FileStation` APIs. Instead of downloading the original photos, it asks the NAS to generate a tiny 10KB thumbnail on the server side and analyzes that instead, saving gigabytes of network bandwidth!

Point it at a shared folder path on your Synology NAS:
```bash
cargo run --release -- /home/Photos/iPhone_backup \
  --nas-host 192.168.1.100 \
  --nas-user myusername \
  --similar \
  --preview
```
*(You will be securely prompted for your DSM Password. If you have 2FA enabled, you will be prompted for your OTP code. If you don't have 2FA enabled, just press Enter to skip).*

## The Interactive Web UI 🖥️
When you run the tool with the `--preview` flag, it instantly spins up a local web server and automatically pops open a glassmorphic dashboard in your default browser. 

In this UI, duplicates are grouped together. The script will automatically pre-select one photo in each group to **KEEP** (usually the highest resolution one) and mark the rest for **DELETE**. You can easily override these selections by clicking on any photo to toggle its status.

**How to pause and resume later?**
Reviewing thousands of photos takes time. If you want to take a break, simply click the **"Save & Exit"** button in the UI. The browser will close your session and the terminal app will gracefully shut down without deleting anything.

Even if you accidentally force-close your browser or your terminal crashes halfway through, **you won't lose your work!** Every single time you click a photo in the UI, your selections are silently auto-saved to a draft file. 

To pick up exactly where you left off, simply run the exact same command but replace `--preview` with `--resume`:
```bash
cargo run --release -- /Users/name/Pictures --resume
```
This will restore your active session and re-open the browser with all of your customized KEEP/DELETE selections perfectly preserved.

## Flags & Options ⚙️

### Common Flags
| Flag | Description |
|---|---|
| `--similar` | Finds visually similar pictures (using perceptual hashing) instead of exact byte-for-byte duplicates. Highly recommended for photo libraries. |
| `--threshold <N>` | The Hamming distance threshold for `--similar`. Default is `10` (out of 127 bits). Set lower (e.g. `5`) to be extremely strict (exact duplicates/burst shots), set higher (e.g. `20`) to catch heavier compressions and resizes. |
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
| `--nas-otp <CODE>` | (Optional) Your 6-digit Synology Secure SignIn app code. If omitted, you will be prompted interactively. If you do not have MFA enabled on your NAS, simply ignore this flag and hit Enter at the interactive prompt. |

*Note: For security, your password is never passed as a flag. The tool will always prompt you interactively.*
