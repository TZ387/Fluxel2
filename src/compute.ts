/* ================================================================
   RUST COMPUTE BRIDGE
   ================================================================
   The physics lives in src-tauri/src/physics/ — this file just calls
   it. Each model exposes two Tauri commands (see models.ts's
   `command` field and lib.rs):
     `<command>_summary(params)` → JSON { derived, valid, reasons },
       cheap regardless of grid size.
     `<command>_volume(params)`  → raw bytes: phi then abs (f32 LE),
       each nx*ny*nz elements of the model's own voxel box (models.ts's
       `grid`, which is not always the parameters — see it), then
       optionally a third nx*ny*nz-byte
       block of per-voxel overlay codes (u8: 0 worst, 1 middling,
       2 best) — sent as a `tauri::ipc::Response` to skip JSON
       serialization of a multi-million-element array. Whether the
       third block is present is self-describing from the buffer's
       length (lib.rs's volume_bytes), since only some models compute
       it (every model but Kubelka-Munk). What the codes *mean* is the
       model's business, not this file's — see models.ts's `overlay`.

   A model that declares `progress` (models.ts) also takes a Channel
   on its volume command and reports its fraction done through it as
   it runs. Only Monte Carlo does: the closed-form models finish
   before a progress readout would render. Either way every command
   is `async` on the Rust side, which is what keeps the run off the
   thread serving the webview.
   ================================================================ */

import { Channel, invoke } from "@tauri-apps/api/core";
import type { ModelDef } from "./models";

export interface RunResult<D = any> {
  phi: Float32Array;
  abs: Float32Array;
  /** Per-voxel overlay codes (0/1/2), or null for models that don't compute one. */
  validity: Uint8Array | null;
  derived: D;
  valid: boolean;
  reasons: string[];
}

export async function runModel<D = any>(
  model: ModelDef<D>,
  params: Record<string, any>,
  onProgress?: (fraction: number) => void
): Promise<RunResult<D>> {
  /* The channel is created only when the caller wants one — a command that
     doesn't declare the argument rejects the call if it's passed anyway. */
  const volumeArgs: Record<string, any> = { params };
  if (onProgress) {
    const channel = new Channel<number>();
    channel.onmessage = onProgress;
    volumeArgs.progress = channel;
  }

  const [summary, raw] = await Promise.all([
    invoke<{ derived: D; valid: boolean; reasons: string[] }>(`${model.command}_summary`, { params }),
    invoke<ArrayBuffer | Uint8Array>(`${model.command}_volume`, volumeArgs),
  ]);

  const bytes = raw instanceof Uint8Array ? raw : new Uint8Array(raw);
  const { nx, ny, nz } = model.grid(params);
  const n = nx * ny * nz;
  /* The grid is worked out twice — here and in Rust — because the buffer has
     to be sliced before anything in it can be read, and for Monte Carlo the
     box is derived from the parameters rather than being them (models.ts's
     `grid`, monte_carlo.rs's display_grid). Checking the length is what
     turns a drift between those two into a legible failure instead of a
     plot of misaligned voxels. */
  if (bytes.byteLength !== n * 8 && bytes.byteLength !== n * 9) {
    throw new Error(
      `${model.command} returned ${bytes.byteLength} bytes for a ${nx}x${ny}x${nz} grid — ` +
        `expected ${n * 8} or ${n * 9}`
    );
  }
  const phi = new Float32Array(bytes.buffer, bytes.byteOffset, n);
  const abs = new Float32Array(bytes.buffer, bytes.byteOffset + n * 4, n);
  const validity = bytes.byteLength > n * 8 ? new Uint8Array(bytes.buffer, bytes.byteOffset + n * 8, n) : null;

  return { phi, abs, validity, derived: summary.derived, valid: summary.valid, reasons: summary.reasons };
}
