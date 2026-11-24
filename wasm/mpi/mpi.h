#ifndef G4_MPI_STUB_H
#define G4_MPI_STUB_H
#include <stdlib.h>
#include <string.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;
typedef int MPI_Info;
typedef struct { int MPI_SOURCE, MPI_TAG, MPI_ERROR; } MPI_Status;
#define MPI_COMM_WORLD 0
#define MPI_INFO_NULL 0
#define MPI_SUCCESS 0
/* datatype handle = element size */
#define MPI_DOUBLE ((MPI_Datatype)sizeof(double))
#define MPI_INT ((MPI_Datatype)sizeof(int))
#define MPI_UNSIGNED_LONG_LONG ((MPI_Datatype)sizeof(unsigned long long))
#define MPI_SUM 1
#define MPI_MAX 2
#define MPI_MIN 3
#define MPI_LAND 4
static inline int MPI_Init(int *a, char ***b) { (void)a; (void)b; return 0; }
static inline int MPI_Finalize(void) { return 0; }
static inline int MPI_Abort(MPI_Comm c, int code) { (void)c; exit(code); return 0; }
static inline int MPI_Comm_rank(MPI_Comm c, int *r) { (void)c; *r = 0; return 0; }
static inline int MPI_Comm_size(MPI_Comm c, int *s) { (void)c; *s = 1; return 0; }
static inline int MPI_Barrier(MPI_Comm c) { (void)c; return 0; }
static inline int MPI_Allreduce(const void *in, void *out, int n, MPI_Datatype t, MPI_Op op, MPI_Comm c)
{ (void)op; (void)c; memcpy(out, in, (size_t)n * (size_t)t); return 0; }
static inline int MPI_Reduce(const void *in, void *out, int n, MPI_Datatype t, MPI_Op op, int root, MPI_Comm c)
{ (void)op; (void)root; (void)c; memcpy(out, in, (size_t)n * (size_t)t); return 0; }
static inline int MPI_Allgather(const void *in, int n, MPI_Datatype t, void *out, int rn, MPI_Datatype rt, MPI_Comm c)
{ (void)rn; (void)rt; (void)c; memcpy(out, in, (size_t)n * (size_t)t); return 0; }
static inline int MPI_Gather(const void *in, int n, MPI_Datatype t, void *out, int rn, MPI_Datatype rt, int root, MPI_Comm c)
{ (void)rn; (void)rt; (void)root; (void)c; memcpy(out, in, (size_t)n * (size_t)t); return 0; }
static inline int MPI_Bcast(void *b, int n, MPI_Datatype t, int root, MPI_Comm c)
{ (void)b; (void)n; (void)t; (void)root; (void)c; return 0; }
static inline int MPI_Send(const void *b, int n, MPI_Datatype t, int dst, int tag, MPI_Comm c)
{ (void)b; (void)n; (void)t; (void)dst; (void)tag; (void)c; abort(); return 1; }
static inline int MPI_Recv(void *b, int n, MPI_Datatype t, int src, int tag, MPI_Comm c, MPI_Status *s)
{ (void)b; (void)n; (void)t; (void)src; (void)tag; (void)c; (void)s; abort(); return 1; }
#ifdef __cplusplus
}
#endif
#endif
