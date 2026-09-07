mod physics;

use physics::fpw1992::{self, Fpw1992Derived, Fpw1992Params};
use physics::kubelka_munk::{self, KubelkaMunkDerived, KubelkaMunkParams};
use physics::liemert_kienle::{self, LiemertKienleDerived, LiemertKienleParams};
use serde::Serialize;
use tauri::ipc::Response;

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
/// pass `Some` here carry it.
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
    volume_bytes(phi, abs, None)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            fpw1992_summary,
            fpw1992_volume,
            kubelka_munk_summary,
            kubelka_munk_volume,
            liemert_kienle_summary,
            liemert_kienle_volume,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
