mod physics;

use physics::fpw1992::{self, Fpw1992Derived, Fpw1992Params};
use physics::kubelka_munk::{self, KubelkaMunkDerived, KubelkaMunkParams};
use physics::liemert_kienle::{self, LiemertKienleDerived, LiemertKienleParams};
use physics::monte_carlo::{self, MonteCarloDerived, MonteCarloParams};
use serde::Serialize;
use tauri::ipc::{Channel, Response};
use tauri::path::BaseDirectory;
use tauri::Manager;

#[derive(Serialize)]
struct Summary<D: Serialize> {
    derived: D,
    valid: bool,
    reasons: Vec<String>,
}

/// phi ++ abs ++ (optional) validity codes: f32 LE for phi/abs, plain u8 for
/// validity. The frontend slices the returned buffer back into typed views
/// at the sizes it already knows (nx*ny*nz each), and tells whether the
/// third part is present from the buffer's total length — only models that
/// pass `Some` here carry it. What the third block *means* is the model's
/// own business: for the diffusion models it grades how well diffusion
/// applies, for Monte Carlo how converged the estimate is (see each
/// model's `overlay` in models.ts, which labels it).
fn volume_bytes(phi: Vec<f32>, abs: Vec<f32>, validity: Option<Vec<u8>>) -> Response {
    let extra = validity.as_ref().map_or(0, |v| v.len());
    let mut bytes = Vec::with_capacity((phi.len() + abs.len()) * 4 + extra);
    for x in &phi {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    for x in &abs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    if let Some(v) = validity {
        bytes.extend_from_slice(&v);
    }
    Response::new(bytes)
}

#[tauri::command(async)]
fn fpw1992_summary(params: Fpw1992Params) -> Summary<Fpw1992Derived> {
    let derived = fpw1992::derived(&params);
    let validity = fpw1992::check_validity(&params, &derived);
    Summary {
        derived,
        valid: validity.valid,
        reasons: validity.reasons,
    }
}

#[tauri::command(async)]
fn fpw1992_volume(params: Fpw1992Params) -> Response {
    let derived = fpw1992::derived(&params);
    let (phi, abs) = fpw1992::compute_volume(&params, &derived);
    let validity = fpw1992::compute_validity_volume(&params, &derived);
    volume_bytes(phi, abs, Some(validity))
}

#[tauri::command(async)]
fn kubelka_munk_summary(params: KubelkaMunkParams) -> Summary<KubelkaMunkDerived> {
    let derived = kubelka_munk::derived(&params);
    let validity = kubelka_munk::check_validity(&params, &derived);
    Summary {
        derived,
        valid: validity.valid,
        reasons: validity.reasons,
    }
}

#[tauri::command(async)]
fn kubelka_munk_volume(params: KubelkaMunkParams) -> Response {
    let (phi, abs) = kubelka_munk::compute_volume(&params);
    volume_bytes(phi, abs, None)
}

#[tauri::command(async)]
fn liemert_kienle_summary(params: LiemertKienleParams) -> Summary<LiemertKienleDerived> {
    let derived = liemert_kienle::derived(&params);
    let validity = liemert_kienle::check_validity(&params, &derived);
    Summary {
        derived,
        valid: validity.valid,
        reasons: validity.reasons,
    }
}

#[tauri::command(async)]
fn liemert_kienle_volume(params: LiemertKienleParams) -> Response {
    let (phi, abs) = liemert_kienle::compute_volume(&params);
    let validity = liemert_kienle::compute_validity_volume(&params);
    volume_bytes(phi, abs, Some(validity))
}

#[tauri::command(async)]
fn monte_carlo_summary(params: MonteCarloParams) -> Summary<MonteCarloDerived> {
    let derived = monte_carlo::derived(&params);
    let validity = monte_carlo::check_validity(&params, &derived);
    Summary {
        derived,
        valid: validity.valid,
        reasons: validity.reasons,
    }
}

/// The only model whose run takes long enough to need saying so while it
/// happens, so it's the only one taking a progress channel. Like every other
/// command here it's declared `async`, which is what puts it on a worker
/// thread rather than the one servicing the webview — that alone is what
/// keeps the window responsive; the channel only gives it something to say.
/// A send that fails means the webview has already dropped the receiver
/// (navigated away, or reloaded mid-run), which is not a reason to abandon
/// the simulation.
#[tauri::command(async)]
fn monte_carlo_volume(params: MonteCarloParams, progress: Channel<f64>) -> Response {
    let (phi, abs, noise) = monte_carlo::compute_volume(&params, |fraction| {
        let _ = progress.send(fraction);
    });
    volume_bytes(phi, abs, Some(noise))
}

/* ================================================================
   SETTINGS FILES
   ================================================================
   Whole-text read and write for the parameter files the frontend saves
   and loads (src/settings.ts owns their shape). No scoping of its own,
   deliberately — the path always comes from a dialog the user just
   picked with, and the webview runs nothing but this app's own bundled
   code, so these are exactly as privileged as the app already is. That's
   also why this exists instead of tauri-plugin-fs, whose scope
   configuration is most of what it would add here.

   Errors are stringly typed because that's what crosses the IPC boundary
   as a rejected promise, and the frontend shows it verbatim.
   ================================================================ */

#[tauri::command(async)]
fn read_text_file(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))
}

#[tauri::command(async)]
fn write_text_file(path: String, contents: String) -> Result<(), String> {
    std::fs::write(&path, contents).map_err(|e| format!("{path}: {e}"))
}

/// Where this install put the bundled example settings files (`resources`
/// in tauri.conf.json). Every install type keeps them somewhere no one
/// would think to browse to — `/usr/lib/fluxel2/examples` for the .deb,
/// the install folder for the Windows installers, a path inside the
/// temporary mount for an AppImage — so without this the files ship but
/// are, in practice, unreachable from the Load dialog.
///
/// `None` rather than an `Err` when there's nothing there: a `npm run
/// tauri dev` run has no bundled resources, and the answer to that is a
/// dialog opening wherever it normally would, not a failure the user has
/// to read. Existence is checked here rather than in the frontend because
/// a `defaultPath` that doesn't exist leaves the dialog nowhere useful.
#[tauri::command(async)]
fn examples_dir(app: tauri::AppHandle) -> Option<String> {
    let dir = app.path().resolve("examples", BaseDirectory::Resource).ok()?;
    dir.is_dir().then(|| dir.to_string_lossy().into_owned())
}

/// Same shape as write_text_file, for contents that aren't valid UTF-8 — a
/// plot's exported PNG, in particular. The frontend hands over the encoded
/// image bytes as a plain array, so this needs no dependency beyond serde's
/// existing Vec<u8> support. Fine at PNG sizes; see write_base64_file for why
/// a volume export goes a different way.
#[tauri::command(async)]
fn write_binary_file(path: String, contents: Vec<u8>) -> Result<(), String> {
    std::fs::write(&path, contents).map_err(|e| format!("{path}: {e}"))
}

/// Standard-alphabet base64, padded or not. Hand-rolled rather than a
/// dependency — see write_base64_file for why this exists at all.
fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    fn sextet(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    let mut group = [0u8; 4];
    let mut filled = 0usize;
    let mut pad = 0usize;
    for c in input.bytes().filter(|c| !c.is_ascii_whitespace()) {
        if c == b'=' {
            pad += 1;
            filled += 1;
        } else {
            group[filled] = sextet(c).ok_or("invalid base64 character")?;
            filled += 1;
        }
        if filled == 4 {
            let n = (group[0] as u32) << 18 | (group[1] as u32) << 12 | (group[2] as u32) << 6 | group[3] as u32;
            out.push((n >> 16) as u8);
            if pad < 2 {
                out.push((n >> 8) as u8);
            }
            if pad < 1 {
                out.push(n as u8);
            }
            group = [0; 4];
            filled = 0;
        }
    }
    Ok(out)
}

/// write_text_file's sibling for binary payloads too large to pass
/// efficiently as write_binary_file's JSON array of numbers — a volume
/// export at the largest grid this app allows (400^3 voxels) is a quarter
/// gigabyte per field, and a JSON number array costs several times that in
/// transit. Base64 costs a third more than the raw bytes instead.
#[tauri::command(async)]
fn write_base64_file(path: String, base64: String) -> Result<(), String> {
    let bytes = decode_base64(&base64).map_err(|e| format!("{path}: {e}"))?;
    std::fs::write(&path, bytes).map_err(|e| format!("{path}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::decode_base64;

    #[test]
    fn decodes_padded_and_unpadded() {
        assert_eq!(decode_base64("").unwrap(), b"");
        assert_eq!(decode_base64("Zg==").unwrap(), b"f");
        assert_eq!(decode_base64("Zm8=").unwrap(), b"fo");
        assert_eq!(decode_base64("Zm9v").unwrap(), b"foo");
        assert_eq!(decode_base64("Zm9vYg==").unwrap(), b"foob");
        assert_eq!(decode_base64("Zm9vYmE=").unwrap(), b"fooba");
        assert_eq!(decode_base64("Zm9vYmFy").unwrap(), b"foobar");
    }

    #[test]
    fn rejects_bad_characters() {
        assert!(decode_base64("not base64!").is_err());
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            fpw1992_summary,
            fpw1992_volume,
            kubelka_munk_summary,
            kubelka_munk_volume,
            liemert_kienle_summary,
            liemert_kienle_volume,
            monte_carlo_summary,
            monte_carlo_volume,
            read_text_file,
            write_text_file,
            examples_dir,
            write_binary_file,
            write_base64_file,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
