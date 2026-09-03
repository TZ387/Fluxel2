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

/// phi ++ abs, each as little-endian f32 — the frontend splits the returned
/// buffer at the midpoint back into two Float32Array views.
fn volume_bytes(phi: Vec<f32>, abs: Vec<f32>) -> Response {
    let mut bytes = Vec::with_capacity((phi.len() + abs.len()) * 4);
    for x in &phi {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    for x in &abs {
        bytes.extend_from_slice(&x.to_le_bytes());
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
    volume_bytes(phi, abs)
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
    volume_bytes(phi, abs)
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
    volume_bytes(phi, abs)
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
