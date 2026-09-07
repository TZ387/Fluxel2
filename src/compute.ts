/* ================================================================
   RUST COMPUTE BRIDGE
   ================================================================
   The physics lives in src-tauri/src/physics/ — this file just calls
   it. Each model exposes two Tauri commands (see models.ts's
   `command` field and lib.rs):
     `<command>_summary(params)` → JSON { derived, valid, reasons },
       cheap regardless of grid size.
     `<command>_volume(params)`  → raw bytes: phi then abs (f32 LE),
       each nx*ny*nz elements, then optionally a third nx*ny*nz-byte
       block of per-voxel validity codes (u8: 0 invalid, 1 marginal,
       2 valid) — sent as a `tauri::ipc::Response` to skip JSON
       serialization of a multi-million-element array. Whether the
       third block is present is self-describing from the buffer's
       length (lib.rs's volume_bytes), since only some models compute
       it (currently FPW1992 only).
   ================================================================ */

import { invoke } from "@tauri-apps/api/core";

export interface RunResult<D = any> {
  phi: Float32Array;
  abs: Float32Array;
  /** Per-voxel validity codes (0/1/2), or null for models that don't compute one. */
  validity: Uint8Array | null;
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
  const validity = bytes.byteLength > n * 8 ? new Uint8Array(bytes.buffer, bytes.byteOffset + n * 8, n) : null;

  return { phi, abs, validity, derived: summary.derived, valid: summary.valid, reasons: summary.reasons };
}
