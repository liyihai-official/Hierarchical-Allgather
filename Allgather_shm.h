#ifndef ALLGATHER_SHM_H
#define ALLGATHER_SHM_H

#include <mpi.h>

#ifdef __cplusplus
extern "C"
{
#endif

  /*
   * Collectively gather into rank-ordered receive blocks, using one shared
   * buffer per node instead of intra-node messages.
   *
   * Requirements, checked on the first call for a (comm, ranks_per_node) pair:
   * - every node holds exactly ranks_per_node consecutive ranks of comm;
   * - datatype is a dense predefined type on homogeneous host memory.
   * A violation returns the same error on every rank: MPI_ERR_TOPOLOGY for the
   * placement, MPI_ERR_TYPE, MPI_ERR_COUNT or MPI_ERR_ARG for the arguments.
   *
   * sendbuf may be MPI_IN_PLACE. The first call for a pair, and a call whose
   * count or datatype differs from the previous one, do collective setup:
   * discard those calls when timing. Other calls allocate nothing.
   * Failures during communication abort the private communicator.
   */
  int Allgather_shm(
    const void* sendbuf,
    int sendcount,
    MPI_Datatype datatype,
    void* recvbuf,
    MPI_Comm comm,
    int ranks_per_node);

  /* Collectively destroy every context created by Allgather_shm. */
  void Allgather_shm_release_cached(void);

  /*
   * Local seconds of the most recent call on this process:
   * [0] write to the shared buffer, and at the root wait for the whole node;
   * [1] send and receive group blocks across nodes;
   * [2] wait for the whole node, then copy to recvbuf.
   */
  extern double Allgather_shm_stage_times[3];

#ifdef __cplusplus
}
#endif

#endif
