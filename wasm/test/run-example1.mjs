// Run Example1 in the wasm module on an in-memory file system and check that
// the live per-step diagnostics arrive and that the output file is written.
// With a reference directory, the live power trace is written next to the
// output for comparison (see compare.py).
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const example = resolve(here, "../../examples/Example1-SteadyState");
const outDir = process.argv[2];

const createModule = (await import(resolve(here, "../build/genesis4.mjs"))).default;
const steps = [];
const M = await createModule({
    print: () => {},
    onG4Step: (step) => steps.push(step),
    g4FieldSlice: -1,
    g4FieldEvery: 100,
});
for (const f of ["Example1.in", "Example1.lat"]) M.FS.writeFile(f, readFileSync(`${example}/${f}`));

const t0 = performance.now();
const rc = M.callMain(["Example1.in"]);
const seconds = (performance.now() - t0) / 1000;
const h5 = M.FS.readFile("Example1.out.h5");

const last = steps.at(-1);
const images = steps.filter((s) => s.field);
console.log(`rc=${rc} run=${seconds.toFixed(2)}s steps=${steps.length}/${last.nz} out=${h5.length} bytes`);
console.log(`keys per step: ${Object.keys(last.data).length}, field images: ${images.length} (${images[0].field.ngrid}x${images[0].field.ngrid})`);
console.log(`power: start ${steps[0].data["Field/power"][0].toExponential(4)} W, end ${last.data["Field/power"][0].toExponential(4)} W at z=${last.z.toFixed(2)} m`);

if (outDir) {
    writeFileSync(`${outDir}/Example1.out.h5`, h5);
    const live = Object.fromEntries(Object.keys(last.data).map((k) => [k, steps.map((s) => s.data[k][0])]));
    writeFileSync(`${outDir}/live.json`, JSON.stringify({ z: steps.map((s) => s.z), ...live }));
}
if (rc !== 0 || steps.length !== last.nz) process.exit(1);
