#ifndef ALLGATHER_RADIX_BATCH_H
#define ALLGATHER_RADIX_BATCH_H

#include <mpi.h>

#ifdef __cplusplus
extern "C"
{
#endif

  typedef struct AllgatherRadixBatchContext AllgatherRadixBatchContext;

  enum
  {
    ALLGATHER_RADIX_BATCH_PORTABLE = 0u,
    ALLGATHER_RADIX_BATCH_NATIVE_FAST = 1u
  };

  /*
   * Collectively create an independent communication context.
   *
   * comm must be a valid intracommunicator. All ranks must pass the same radix
   * (>= 2) and ranks_per_group (>= 1, dividing the communicator size).
   * Groups consist of consecutive communicator ranks; no physical topology is
   * inferred. On a recoverable setup error, every rank returns the same error
   * and every valid output pointer is set to NULL.
   */
  int Allgather_radix_batch_context_create(
    MPI_Comm comm,
    int radix,
    int ranks_per_group,
    AllgatherRadixBatchContext** context);

  /*
   * Collectively bind counts/types and reserve storage once, outside a hot loop.
   * A failed prepare preserves the previous successful plan. The bound datatype
   * handles must remain valid until a new prepare or destruction.
   *
   * ALLGATHER_RADIX_BATCH_NATIVE_FAST permits MPI_BYTE transfers of dense host-memory layouts on
   * a homogeneous machine. Only MPI named types with proven dense bounds use
   * that path; derived types fall back even when their extent equals their size
   * (their typemap may reverse element order). ALLGATHER_RADIX_BATCH_PORTABLE
   * always preserves independent MPI packing units.
   *
   * After prepare, Allgather_radix_batch_ctx performs no allocation, datatype
   * queries, communicator duplication or validation collectives. Its arguments
   * must match the plan (in-place still ignores sendcount/sendtype). Invalid
   * bindings or execution errors abort rather than strand peers. Buffer addresses
   * may change between calls. To change counts/types/mode, prepare again.
   */
  int Allgather_radix_batch_prepare(
    AllgatherRadixBatchContext* context,
    int sendcount,
    MPI_Datatype sendtype,
    int recvcount,
    MPI_Datatype recvtype,
    int in_place,
    unsigned options);

  /*
   * Collectively gather into rank-ordered receive blocks.
   *
   * Like MPI_Allgather, each sender's type signature must match each receiver's
   * (recvcount, recvtype) signature. Layouts and counts may differ when those
   * signatures agree. Signatures are a caller contract; byte-size checks cannot
   * prove full signature equality.
   *
   * Supported: noncontiguous/resized types with nonnegative extent, negative
   * typemap displacements, empty payloads, and MPI_IN_PLACE at all ranks.
   * In-place calls ignore sendcount/sendtype entirely. MPI_BOTTOM/absolute-address
   * buffers and intercommunicators are unsupported. The caller supplies storage
   * covering all datatype displacements; receive contributions must not overlap.
   *
   * Valid context/datatype handles and matching collective call order are caller
   * preconditions. A context is not reentrant: serialize calls and destruction.
   * Each wire message uses an int MPI count; oversized layouts return
   * MPI_ERR_COUNT rather than overflowing. Workspace is retained until destroy.
   *
   * Before prepare, argument/allocation/pack/unpack errors are agreed at safe
   * collective boundaries. After prepare, the documented execution contract
   * applies. Unexpected internal communication errors abort the private
   * communicator: portable MPI cannot promise recovery after such a failure.
   * Receive contents are unspecified after any non-success return.
   */
  int Allgather_radix_batch_ctx(
    AllgatherRadixBatchContext* context,
    const void* sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    void* recvbuf,
    int recvcount,
    MPI_Datatype recvtype);

  /* Collectively release the private communicator and all owned workspace. */
  int Allgather_radix_batch_context_destroy(AllgatherRadixBatchContext** context);

  /*
   * Convenience entry point matching the original algorithm's argument order.
   * sendcount/datatype also specify each received contribution, even in-place.
   * The first call for a (comm, radix, ranks_per_group) creates and caches a
   * context (collective); a call whose count/datatype/in-place binding differs
   * from the previous one on that context prepares it again (collective).
   * Other calls only communicate. Discard those first calls when timing.
   * The cache is keyed by the comm handle: call
   * Allgather_radix_batch_release_cached() before freeing a communicator that
   * was used here. Native dense transfers require homogeneous host memory.
   */
  int Allgather_radix_batch(
    const void* sendbuf,
    int sendcount,
    MPI_Datatype datatype,
    void* recvbuf,
    MPI_Comm comm,
    int radix,
    int ranks_per_group);

  /* Collectively destroy every context cached by Allgather_radix_batch. */
  void Allgather_radix_batch_release_cached(void);

#ifdef __cplusplus
}
#endif

#endif