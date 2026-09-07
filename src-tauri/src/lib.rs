mod physics;

use physics::fpw1992::{self, Fpw1992Derived, Fpw1992Params};
use physics::kubelka_munk::{self, KubelkaMunkDerived, KubelkaMunkParams};
use physics::liemert_kienle::{self, LiemertKienleDerived, LiemertKienleParams};
use physics::monte_carlo::{self, MonteCarloDerived, MonteCarloParams};
use serde::Serialize;
use tauri::ipc::{Channel, Response};

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
   Whole-text read and write, for the parameter files the frontend
   saves and loads (src/settings.ts owns their shape). The path always
   comes from the native dialog the user has just picked with, so this
   pair deliberately does no scoping of its own — which is the reason
   it exists instead of tauri-plugin-fs, whose scope configuration is
   most of what that plugin would add here. The webview runs nothing
   but this app's own bundled code, so these are exactly as privileged
   as the app already is, and no more.

   The error is stringly typed because that is what crosses the IPC
   boundary as a rejected promise; the frontend shows it verbatim, so
   it carries the path as well as the reason.
   ================================================================ */

#[tauri::command(async)]
fn read_text_file(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))
}

#[tauri::command(async)]
fn write_text_file(path: String, contents: String) -> Result<(), String> {
    std::fs::write(&path, contents).map_err(|e| format!("{path}: {e}"))
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
