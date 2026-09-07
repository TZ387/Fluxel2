/* ================================================================
   FRONTEND TEST RUNNER
   ================================================================
   `npm test`. Bundles every tests/*.test.ts with esbuild — which is
   what resolves the extensionless imports the src/ files use, the
   same way Vite does in the app — and runs each in its own node
   process.

   Separate processes on purpose: each test installs a global `document`
   stub, and one process per file keeps those from seeing each other,
   while a file that throws outright is reported rather than taking the
   runner down with it.

   No test framework. There are two test files, they need a counter and
   a non-zero exit (see harness.ts), and a framework would add a
   dependency for nothing they use.
   ================================================================ */
import { build } from "esbuild";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const tests = readdirSync(here)
  .filter((f) => f.endsWith(".test.ts"))
  .sort();

if (tests.length === 0) {
  console.error("no tests/*.test.ts found");
  process.exit(1);
}

const out = mkdtempSync(join(tmpdir(), "fluxel-tests-"));
let failed = 0;

try {
  for (const test of tests) {
    const bundle = join(out, test.replace(/\.ts$/, ".mjs"));
    await build({
      entryPoints: [join(here, test)],
      outfile: bundle,
      bundle: true,
      platform: "node",
      format: "esm",
      target: "node20",
      logLevel: "warning",
    });
    console.log(`\n── ${test} ${"─".repeat(Math.max(0, 60 - test.length))}`);
    const run = spawnSync(process.execPath, [bundle], { stdio: "inherit" });
    if (run.status !== 0) failed++;
  }
} finally {
  rmSync(out, { recursive: true, force: true });
}

console.log(
  failed === 0
    ? `\n${tests.length} test file(s) passed`
    : `\n${failed} of ${tests.length} test file(s) FAILED`
);
process.exit(failed === 0 ? 0 : 1);
