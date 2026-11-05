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

  /*
   * Same exchange as Allgather_shm, without the final copy: *result points to the
   * contributions of all ranks in rank order, in one buffer shared by the node.
   * The requirements above apply. *result is NULL when the contributions are empty.
   *
   * This is not a drop-in MPI_Allgather:
   * - the result is mapped read-only, so a write faults;
   * - it stays valid until this process makes its next Allgather_shm or
   *   Allgather_shm_view call for the same (comm, ranks_per_node), or calls
   *   Allgather_shm_release_cached. To change or keep it, copy it first;
   * - that next call waits for every process of the node to make its next call,
   *   because the buffer is then reused;
   * - sendbuf must not point into the previous result.
   * MPI_IN_PLACE returns MPI_ERR_BUFFER and a NULL result returns MPI_ERR_ARG; neither
   * communicates, so every rank must make the same mistake to get the same error.
   */
  int Allgather_shm_view(
    const void* sendbuf,
    int sendcount,
    MPI_Datatype datatype,
    const void** result,
    MPI_Comm comm,
    int ranks_per_node);

  /* Collectively destroy every context created by Allgather_shm or Allgather_shm_view. */
  void Allgather_shm_release_cached(void);

  /*
   * Local seconds of the most recent call on this process:
   * [0] write to the shared buffer, and at the root wait for the whole node;
   * [1] send and receive group blocks across nodes;
   * [2] wait for the whole node, then copy to recvbuf (Allgather_shm_view only waits).
   */
  extern double Allgather_shm_stage_times[3];

#ifdef __cplusplus
}
#endif

#endif
