#include <mpi.h>
#include <hdf5.h>
#ifndef H5_HAVE_PARALLEL
#define H5FD_MPIO_INDEPENDENT 0
#define H5FD_MPIO_COLLECTIVE 1
static inline herr_t H5Pset_fapl_mpio(hid_t p, MPI_Comm c, MPI_Info i) { (void)p; (void)c; (void)i; return 0; }
static inline herr_t H5Pset_dxpl_mpio(hid_t p, int m) { (void)p; (void)m; return 0; }
#endif
