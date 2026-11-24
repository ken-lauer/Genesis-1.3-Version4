// Live diagnostics for the WebAssembly build: after every output step the
// values just computed are handed to a JavaScript callback, so a page can plot
// a run while it is in progress instead of waiting for the output file.
#ifdef __EMSCRIPTEN__

#include <emscripten.h>

#include "Diagnostic.h"
#include "Field.h"

// Returns the field slice wanted this step: >= 0 a slice, -1 the slice with
// the highest power, -2 none. Module.onG4Step unset turns the hook off (-3).
EM_JS(int, g4_live_begin, (int iz, int nz, int ns, double z), {
    if (!Module.onG4Step) return -3;
    Module.g4Step = {iz, nz, ns, z, data: {}, field: null};
    const slice = Module.g4FieldSlice;
    const every = Module.g4FieldEvery || 1;
    return slice === undefined || iz % every ? -2 : slice;
});

EM_JS(void, g4_live_data, (const char *group, const char *key, const double *p, int n), {
    const name = UTF8ToString(group) + "/" + UTF8ToString(key);
    if (Module.g4Keys && !Module.g4Keys.has(name)) return;
    Module.g4Step.data[name] = HEAPF64.slice(p >> 3, (p >> 3) + n);
});

// p: ngrid*ngrid complex values as interleaved (re, im) doubles on a grid
// spanning +-gridmax
EM_JS(void, g4_live_field, (int slice, int ngrid, double gridmax, const double *p), {
    Module.g4Step.field = {slice, ngrid, gridmax,
                           data: HEAPF64.slice(p >> 3, (p >> 3) + 2 * ngrid * ngrid)};
});

EM_JS(void, g4_live_end, (), {
    const step = Module.g4Step;
    Module.g4Step = null;
    Module.onG4Step(step);
});

void Diagnostic::emitLive(std::vector<Field *> *field) const
{
    const int step = iz - 1;  // calc() has already advanced iz
    if (step < 0) { return; }
    const int want = g4_live_begin(step, nz, ns, zout[step]);
    if (want == -3) { return; }

    const size_t per_step = static_cast<size_t>(ns);
    const size_t all_steps = per_step * static_cast<size_t>(nz);
    for (size_t igrp = 2; igrp < val.size(); igrp++) {
        std::string group = "Beam";
        if (igrp > 2) {
            const int harm = field->at(igrp - 3)->getHarm();
            group = harm > 1 ? "Field" + std::to_string(harm) : "Field";
        }
        for (auto const &[key, data] : val[igrp]) {
            // only the per-slice, per-step records change from step to step
            if (data.size() != all_steps) { continue; }
            g4_live_data(group.c_str(), key.c_str(), &data[step * per_step], ns);
        }
    }

    if (want >= -1 && !field->empty()) {
        Field *fld = field->at(0);
        int slice = want;
        if (slice < 0 || slice >= ns) {
            slice = 0;
            auto const it = val[3].find("power");
            if (it != val[3].end()) {
                const double *pw = &it->second[step * per_step];
                for (int is = 1; is < ns; is++) {
                    if (pw[is] > pw[slice]) { slice = is; }
                }
            }
        }
        const size_t stored = (static_cast<size_t>(slice) + fld->first) % fld->field.size();
        g4_live_field(slice, fld->ngrid, fld->gridmax,
                      reinterpret_cast<const double *>(fld->field[stored].data()));
    }
    g4_live_end();
}

#endif
