#include <fftw3.h>
/* FFTW has no cycle counter on wasm and falls back to a wall-clock timer that
   needs 10 ms per candidate plan, so FFTW_MEASURE planning takes many minutes
   (worse in browsers, which coarsen timers). Plan by heuristic instead; this
   also makes the chosen plan, and so the results, reproducible. */
#undef FFTW_MEASURE
#define FFTW_MEASURE FFTW_ESTIMATE
