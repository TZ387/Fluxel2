/* ================================================================
   RUST COMPUTE BRIDGE
   ================================================================
   The physics lives in src-tauri/src/physics/ — this file just calls
   it. Each model exposes two Tauri commands (see models.ts's
   `command` field and lib.rs):
     `<command>_summary(params)` → JSON { derived, valid, reasons },
       cheap regardless of grid size.
     `<command>_volume(params)`  → raw bytes: phi then abs (f32 LE),
       each nx*ny*nz elements — sent as a `tauri::ipc::Response` to
       skip JSON serialization of a multi-million-element array.
   ================================================================ */

import { invoke } from "@tauri-apps/api/core";

export interface RunResult<D = any> {
  phi: Float32Array;
  abs: Float32Array;
  derived: D;
  valid: boolean;
  reasons: string[];
}

export async function runModel<D = any>(
  command: string,
  params: Record<string, any>
): Promise<RunResult<D>> {
  const [summary, raw] = await Promise.all([
    invoke<{ derived: D; valid: boolean; reasons: string[] }>(`${command}_summary`, { params }),
    invoke<ArrayBuffer | Uint8Array>(`${command}_volume`, { params }),
  ]);

  const bytes = raw instanceof Uint8Array ? raw : new Uint8Array(raw);
  const n = (params.nx as number) * (params.ny as number) * (params.nz as number);
  const phi = new Float32Array(bytes.buffer, bytes.byteOffset, n);
  const abs = new Float32Array(bytes.buffer, bytes.byteOffset + n * 4, n);

  return { phi, abs, derived: summary.derived, valid: summary.valid, reasons: summary.reasons };
}
