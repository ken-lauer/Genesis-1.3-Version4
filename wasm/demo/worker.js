// One Genesis4 run per worker: the module keeps global state between runs, and
// terminating the worker is the only way to cancel a run.
import createModule from "./genesis4.mjs";

const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

function intensity({ ngrid, data }) {
    const out = new Float32Array(ngrid * ngrid);
    for (let i = 0; i < out.length; i++) out[i] = data[2 * i] ** 2 + data[2 * i + 1] ** 2;
    return out;
}

self.onmessage = async ({ data: { files, input, fieldEvery } }) => {
    const log = (line) => postMessage({ type: "log", line });
    const M = await createModule({
        print: log,
        printErr: log,
        g4Keys: new Set(["Field/power", "Beam/bunching"]),
        g4FieldSlice: -1,
        g4FieldEvery: fieldEvery,
        onG4Step: ({ iz, nz, ns, z, data, field }) => {
            const power = data["Field/power"];
            const msg = { type: "step", iz, nz, ns, z, power, meanPower: mean(power), meanBunching: mean(data["Beam/bunching"]) };
            const transfer = [power.buffer];
            if (field) {
                msg.image = { slice: field.slice, ngrid: field.ngrid, gridmax: field.gridmax, data: intensity(field) };
                transfer.push(msg.image.data.buffer);
            }
            postMessage(msg, transfer);
        },
    });
    for (const [name, text] of Object.entries(files)) M.FS.writeFile(name, text);
    const t0 = performance.now();
    const rc = M.callMain([input]);
    const seconds = (performance.now() - t0) / 1000;
    const name = M.FS.readdir("/").find((f) => f.endsWith(".out.h5"));
    const output = name ? M.FS.readFile(name) : null;
    postMessage({ type: "done", rc, seconds, name, output }, output ? [output.buffer] : []);
};
