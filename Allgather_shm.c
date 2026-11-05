/**
 * @file Allgather_shm.c
 * @brief 节点共享内存版 Allgather：节点内不传消息，只有第 2 步跨节点通信。
 *
 * @details
 * 记号与 Allgather_radix_batch.c 相同：P 个进程，每组 B 个连续 rank，G = P / B 组，这里
 * 要求一组正好是一个节点。m 为每个进程贡献的字节数，组块 D_g 为组 g 的 B × m 字节。
 *
 * 每个节点有一块 P × m 字节的共享缓冲区 data，rank r 的贡献固定放在 r × m 处，所以写入后
 * 不需要重排。一次调用分三步：
 * 1. 每个进程把自己的贡献写入 data；组 g 的根（本地 rank g mod B）等本节点全部写完。
 * 2. 与 distribute_group_blocks() 相同的分批跨节点分发，只是根直接从 data 发送，接收也
 *    直接写入 data。
 * 3. 等本节点所有进程完成第 2 步，每个进程把 data 拷到自己的 recvbuf。
 *
 * 同步靠 data 前面的三个单调递增计数器。第 n 次调用中，计数到达 n × B 就表示本节点所有
 * 进程都完成了相应的一步，因此不需要复位。下一次调用开始时先等 copied 到位，这样较快的
 * 进程不会覆盖较慢的进程还在读的数据。
 *
 * Allgather_shm_view() 不做第 3 步的拷贝，每个进程通过只读映射直接读 data。它的 copied
 * 在该进程下一次调用开始时才加一，所以数据要等本节点所有进程都进入下一次调用才会被覆盖。
 *
 * 缓冲区用 /dev/shm 文件映射，而不是 MPI_Win_allocate_shared：多节点时 MPICH 创建窗口会对
 * 整块内存做 NIC 注册，页面全部落在节点内 rank 0 所在的 NUMA 节点上，无法再安排位置。
 */
#define _POSIX_C_SOURCE 200809L

#include "Allgather_shm.h"

#include <fcntl.h>
#include <limits.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/* 计数器放在多个进程共享的内存里，必须无锁才能跨进程使用。 */
_Static_assert(ATOMIC_LLONG_LOCK_FREE == 2, "Shared counters must be lock-free");

enum
{
  ALLGATHER_SHM_TAG_DISTRIBUTE = 1,      ///< 第 2 步的消息标签
  ALLGATHER_SHM_PROGRESS_INTERVAL = 1024 ///< 自旋等待时，每多少次轮询推进一次 MPI
};

double Allgather_shm_stage_times[3];

/** @brief 节点共享的计数器，各占一个缓存行，避免互相干扰。 */
typedef struct
{
  _Alignas(64) atomic_ullong written;   ///< 第 1 步：已写入 data 的进程累计数
  _Alignas(64) atomic_ullong received;  ///< 第 2 步：已完成跨节点收发的进程累计数
  _Alignas(64) atomic_ullong copied;    ///< 第 3 步：已不再读取 data 的进程累计数
} NodeCounters;

/** @brief 一个 (comm, B) 对应的上下文。 */
typedef struct AllgatherShmContext
{
  struct AllgatherShmContext* next;  ///< 缓存链表的下一项
  MPI_Comm user_comm;                ///< 缓存键：用户的通信子
  MPI_Comm comm;                     ///< 私有通信子
  MPI_Comm node_comm;                ///< 本节点的通信子
  unsigned char* mapping;            ///< 共享映射的首地址；尚未分配时为 NULL
  size_t mapping_bytes;              ///< 共享映射的字节数
  NodeCounters* counters;            ///< 映射内的计数器（第一页）
  unsigned char* data;               ///< 映射内的数据区（第二页起）
  unsigned char* read_only_mapping;  ///< 同一块内存的只读映射的首地址
  const unsigned char* read_only_data; ///< 只读映射内的数据区，交给 Allgather_shm_view() 的调用者
  int result_held;                   ///< 非 0 表示本进程还持有上一次 Allgather_shm_view() 的结果
  size_t capacity_bytes;             ///< data 的容量（字节）
  size_t record_bytes;               ///< m：每个进程贡献的字节数
  size_t group_bytes;                ///< 组块字节数 = B × m
  size_t data_bytes;                 ///< 本次绑定使用的字节数 = P × m
  unsigned long long call_index;     ///< 当前映射上已开始的调用次数
  MPI_Request* requests;             ///< 第 2 步的请求数组，共 max(G − 1, 1) 个
  MPI_Datatype bound_type;           ///< 绑定的数据类型
  int bound_count;                   ///< 绑定的元素个数；-1 表示尚未绑定
  int rank;                          ///< 本进程在 comm 中的 rank
  int process_count;                 ///< P
  int ranks_per_node;                ///< B
  int group_index;                   ///< 组号 = rank / B
  int local_rank;                    ///< 本地 rank = rank mod B
  int group_count;                   ///< G
} AllgatherShmContext;

static AllgatherShmContext* cached_contexts = NULL;

/** @brief 通信进行中的错误无法安全恢复，直接中止。 */
static void require_ok(MPI_Comm comm, int error)
{
  if (error != MPI_SUCCESS)
  {
    MPI_Abort(comm, error);
  }
}

static void cpu_relax(void)
{
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#endif
}

/**
 * @brief 自旋等待计数器到达 target。
 *
 * 等待期间周期性地调用 MPI_Iprobe 推进 MPI，这样用户在调用前发起的非阻塞通信不会因为本
 * 进程只在自旋而无法完成。
 */
static void spin_until(const AllgatherShmContext* context,
                       atomic_ullong* counter,
                       unsigned long long target)
{
  int polls = 0;

  while (atomic_load_explicit(counter, memory_order_acquire) < target)
  {
    cpu_relax();
    if (++polls == ALLGATHER_SHM_PROGRESS_INTERVAL)
    {
      int flag;
      polls = 0;
      require_ok(context->comm,
                 MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, context->comm, &flag, MPI_STATUS_IGNORE));
    }
  }
}

/** @brief 判断类型是否为布局稠密的预定义类型（判据与 Allgather_radix_batch.c 相同）。 */
static int is_dense_type(MPI_Datatype type)
{
  int integers, addresses, datatypes, combiner, bytes;
  MPI_Aint lower_bound, extent, true_lower_bound, true_extent;

  if (MPI_Type_get_envelope(type, &integers, &addresses, &datatypes, &combiner) != MPI_SUCCESS ||
      combiner != MPI_COMBINER_NAMED || MPI_Type_size(type, &bytes) != MPI_SUCCESS ||
      MPI_Type_get_extent(type, &lower_bound, &extent) != MPI_SUCCESS ||
      MPI_Type_get_true_extent(type, &true_lower_bound, &true_extent) != MPI_SUCCESS)
  {
    return 0;
  }
  return lower_bound == 0 && true_lower_bound == 0 && extent == bytes && true_extent == bytes;
}

/** @brief 本进程不再读上一次 Allgather_shm_view() 的结果，此后 data 可以被覆盖。 */
static void release_result(AllgatherShmContext* context)
{
  if (context->result_held)
  {
    context->result_held = 0;
    atomic_fetch_add_explicit(&context->counters->copied, 1, memory_order_release);
  }
}

/** @brief 释放共享映射。集合操作；等本节点所有进程都读完上一次调用的数据。 */
static void free_shared(AllgatherShmContext* context)
{
  if (context->mapping != NULL)
  {
    release_result(context);
    spin_until(context,
               &context->counters->copied,
               context->call_index * (unsigned long long)context->ranks_per_node);
    require_ok(context->comm, MPI_Barrier(context->node_comm));
    munmap(context->mapping, context->mapping_bytes);
    munmap(context->read_only_mapping, context->mapping_bytes);
    context->mapping = NULL;
    context->read_only_mapping = NULL;
    context->read_only_data = NULL;
    context->counters = NULL;
    context->data = NULL;
    context->capacity_bytes = 0;
    context->call_index = 0;
  }
}

static void destroy_context(AllgatherShmContext* context)
{
  free_shared(context);
  require_ok(context->comm, MPI_Comm_free(&context->node_comm));
  require_ok(context->comm, MPI_Comm_free(&context->comm));
  free(context->requests);
  free(context);
}

/**
 * @brief 集合地创建上下文，并确认每个节点正好是 B 个连续 rank。
 * @return MPI_SUCCESS；B 不合法时返回 MPI_ERR_ARG；放置不符合要求时返回 MPI_ERR_TOPOLOGY。
 *         所有进程返回相同的结果。
 */
static int create_context(MPI_Comm comm, int ranks_per_node, AllgatherShmContext** result)
{
  AllgatherShmContext* context;
  int rank, process_count, node_size, node_min, node_max, valid;

  require_ok(comm, MPI_Comm_rank(comm, &rank));
  require_ok(comm, MPI_Comm_size(comm, &process_count));
  valid = ranks_per_node >= 1 && process_count % ranks_per_node == 0;
  require_ok(comm, MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, comm));
  if (!valid)
  {
    return MPI_ERR_ARG;
  }

  context = calloc(1, sizeof(*context));
  if (context == NULL)
  {
    require_ok(comm, MPI_ERR_NO_MEM);
  }
  context->user_comm = comm;
  context->bound_count = -1;
  context->rank = rank;
  context->process_count = process_count;
  context->ranks_per_node = ranks_per_node;
  context->group_index = rank / ranks_per_node;
  context->local_rank = rank % ranks_per_node;
  context->group_count = process_count / ranks_per_node;
  context->requests = malloc((size_t)(context->group_count > 1 ? context->group_count - 1 : 1) *
                             sizeof(*context->requests));
  if (context->requests == NULL)
  {
    require_ok(comm, MPI_ERR_NO_MEM);
  }

  require_ok(comm, MPI_Comm_dup(comm, &context->comm));
  require_ok(comm,
             MPI_Comm_split_type(
               context->comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &context->node_comm));
  require_ok(comm, MPI_Comm_size(context->node_comm, &node_size));
  require_ok(comm, MPI_Allreduce(&rank, &node_min, 1, MPI_INT, MPI_MIN, context->node_comm));
  require_ok(comm, MPI_Allreduce(&rank, &node_max, 1, MPI_INT, MPI_MAX, context->node_comm));
  valid = node_size == ranks_per_node && node_min == context->group_index * ranks_per_node &&
          node_max == node_min + ranks_per_node - 1;
  require_ok(comm, MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, context->comm));
  if (!valid)
  {
    destroy_context(context);
    return MPI_ERR_TOPOLOGY;
  }
  *result = context;
  return MPI_SUCCESS;
}

/**
 * @brief 建立节点共享映射，并把数据页轮流分给本节点各进程首次写入。
 *
 * Linux 把页面放在首次写入它的进程所在的 NUMA 节点上。轮流写入让页面均匀分布在各 CPU
 * 插槽上，第 3 步的跨插槽读取就不会集中到一个插槽的内存上。所有进程映射完后立即删除
 * 文件名，进程退出时内存自动释放。
 */
static void allocate_shared(AllgatherShmContext* context, size_t data_bytes)
{
  static unsigned segment_serial = 0;
  size_t page_bytes = (size_t)sysconf(_SC_PAGESIZE);
  size_t pages = (data_bytes + page_bytes - 1) / page_bytes;
  char path[80] = {0};
  int fd;

  context->mapping_bytes = (pages + 1) * page_bytes;
  if (context->local_rank == 0)
  {
    snprintf(path, sizeof(path), "/dev/shm/allgather_shm_%ld_%u", (long)getpid(), segment_serial++);
    fd = open(path, O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0 || ftruncate(fd, (off_t)context->mapping_bytes) != 0)
    {
      require_ok(context->comm, MPI_ERR_NO_MEM);
    }
  }
  require_ok(context->comm, MPI_Bcast(path, (int)sizeof(path), MPI_CHAR, 0, context->node_comm));
  if (context->local_rank != 0)
  {
    fd = open(path, O_RDWR);
    if (fd < 0)
    {
      require_ok(context->comm, MPI_ERR_NO_MEM);
    }
  }
  context->mapping =
    mmap(NULL, context->mapping_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  context->read_only_mapping = mmap(NULL, context->mapping_bytes, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (context->mapping == MAP_FAILED || context->read_only_mapping == MAP_FAILED)
  {
    require_ok(context->comm, MPI_ERR_NO_MEM);
  }
  context->counters = (NodeCounters*)context->mapping;
  context->data = context->mapping + page_bytes;
  context->read_only_data = context->read_only_mapping + page_bytes;
  context->capacity_bytes = data_bytes;
  context->call_index = 0;

  if (context->local_rank == 0)
  {
    atomic_init(&context->counters->written, 0);
    atomic_init(&context->counters->received, 0);
    atomic_init(&context->counters->copied, 0);
  }
  for (size_t page = (size_t)context->local_rank; page < pages;
       page += (size_t)context->ranks_per_node)
  {
    context->data[page * page_bytes] = 0;
  }
  require_ok(context->comm, MPI_Barrier(context->node_comm));
  if (context->local_rank == 0)
  {
    unlink(path);
  }
}

/**
 * @brief 集合地绑定个数和类型，并按需扩大共享映射。
 * @return MPI_SUCCESS；类型不稠密返回 MPI_ERR_TYPE；个数不一致返回 MPI_ERR_ARG；组块超出
 *         int 范围返回 MPI_ERR_COUNT。所有进程返回相同的结果。
 */
static int prepare(AllgatherShmContext* context, int count, MPI_Datatype type)
{
  int bytes = 0;
  long long extremes[2];
  int status[2];
  long long record_bytes;

  status[0] = count >= 0 && is_dense_type(type);
  require_ok(context->comm, MPI_Type_size(type, &bytes));
  record_bytes = (long long)count * bytes;
  extremes[0] = record_bytes;
  extremes[1] = -record_bytes;
  require_ok(context->comm,
             MPI_Allreduce(MPI_IN_PLACE, extremes, 2, MPI_LONG_LONG, MPI_MIN, context->comm));
  status[1] = extremes[0] == -extremes[1];
  require_ok(context->comm, MPI_Allreduce(MPI_IN_PLACE, status, 2, MPI_INT, MPI_MIN, context->comm));
  if (!status[0])
  {
    return MPI_ERR_TYPE;
  }
  if (!status[1])
  {
    return MPI_ERR_ARG;
  }
  if (record_bytes > INT_MAX / context->ranks_per_node)
  {
    return MPI_ERR_COUNT;
  }

  context->record_bytes = (size_t)record_bytes;
  context->group_bytes = context->record_bytes * (size_t)context->ranks_per_node;
  context->data_bytes = context->record_bytes * (size_t)context->process_count;
  if (context->data_bytes > context->capacity_bytes)
  {
    free_shared(context);
    allocate_shared(context, context->data_bytes);
  }
  context->bound_count = count;
  context->bound_type = type;
  return MPI_SUCCESS;
}

/**
 * @brief 第 2 步：把组块分发到其他节点，分批方式与 distribute_group_blocks() 相同。
 *
 * 第 s 批中，组 s·B + t 的根（本地 rank t）把组块发给其他每个组的本地 rank t。数据都在
 * data 的最终位置，所以根不需要拷贝。
 */
static void distribute(AllgatherShmContext* context)
{
  int batch_begin = 0;

  while (batch_begin < context->group_count)
  {
    int remaining = context->group_count - batch_begin;
    int batch_groups = remaining < context->ranks_per_node ? remaining : context->ranks_per_node;
    int request_count = 0;

    if (context->local_rank < batch_groups)
    {
      int source_group = batch_begin + context->local_rank;
      int source_rank = source_group * context->ranks_per_node + context->local_rank;
      unsigned char* block = context->data + (size_t)source_group * context->group_bytes;

      if (context->rank == source_rank)
      {
        for (int group = 0; group < context->group_count; ++group)
        {
          if (group != source_group)
          {
            require_ok(context->comm,
                       MPI_Isend(block,
                                 (int)context->group_bytes,
                                 MPI_BYTE,
                                 group * context->ranks_per_node + context->local_rank,
                                 ALLGATHER_SHM_TAG_DISTRIBUTE,
                                 context->comm,
                                 &context->requests[request_count++]));
          }
        }
      }
      else
      {
        require_ok(context->comm,
                   MPI_Irecv(block,
                             (int)context->group_bytes,
                             MPI_BYTE,
                             source_rank,
                             ALLGATHER_SHM_TAG_DISTRIBUTE,
                             context->comm,
                             &context->requests[request_count++]));
      }
    }
    require_ok(context->comm,
               MPI_Waitall(request_count, context->requests, MPI_STATUSES_IGNORE));
    batch_begin += batch_groups;
  }
}

/**
 * @brief 第 1、2 步，以及第 3 步开头的等待：返回时本节点的 data 已经完整。
 * @return 本进程完成第 2 步的时刻，第 3 步的耗时从这里算起。
 */
static double exchange(AllgatherShmContext* context, const void* own_data)
{
  NodeCounters* counters = context->counters;
  unsigned long long node_ranks = (unsigned long long)context->ranks_per_node;
  size_t own_offset = (size_t)context->rank * context->record_bytes;
  unsigned long long target;
  double t0, t1, t2;

  t0 = MPI_Wtime();
  release_result(context);
  spin_until(context, &counters->copied, context->call_index * node_ranks);
  target = ++context->call_index * node_ranks;

  memcpy(context->data + own_offset, own_data, context->record_bytes);
  atomic_fetch_add_explicit(&counters->written, 1, memory_order_release);
  if (context->local_rank == context->group_index % context->ranks_per_node)
  {
    spin_until(context, &counters->written, target);
  }
  t1 = MPI_Wtime();

  distribute(context);
  atomic_fetch_add_explicit(&counters->received, 1, memory_order_release);
  t2 = MPI_Wtime();

  spin_until(context, &counters->received, target);
  Allgather_shm_stage_times[0] = t1 - t0;
  Allgather_shm_stage_times[1] = t2 - t1;
  return t2;
}

/** @brief 找到或创建 (comm, B) 的上下文，并绑定个数和类型。 */
static int bind_context(int sendcount,
                        MPI_Datatype datatype,
                        MPI_Comm comm,
                        int ranks_per_node,
                        AllgatherShmContext** result)
{
  AllgatherShmContext* context = cached_contexts;
  int error;

  while (context != NULL &&
         (context->user_comm != comm || context->ranks_per_node != ranks_per_node))
  {
    context = context->next;
  }
  if (context == NULL)
  {
    error = create_context(comm, ranks_per_node, &context);
    if (error != MPI_SUCCESS)
    {
      return error;
    }
    context->next = cached_contexts;
    cached_contexts = context;
  }
  if (context->bound_count != sendcount || context->bound_type != datatype)
  {
    error = prepare(context, sendcount, datatype);
    if (error != MPI_SUCCESS)
    {
      context->bound_count = -1;
      return error;
    }
  }
  *result = context;
  return MPI_SUCCESS;
}

int Allgather_shm(const void* sendbuf,
                  int sendcount,
                  MPI_Datatype datatype,
                  void* recvbuf,
                  MPI_Comm comm,
                  int ranks_per_node)
{
  AllgatherShmContext* context;
  int error = bind_context(sendcount, datatype, comm, ranks_per_node, &context);

  if (error != MPI_SUCCESS)
  {
    return error;
  }
  if (context->data_bytes > 0)
  {
    size_t own_offset = (size_t)context->rank * context->record_bytes;
    const void* own_data =
      sendbuf == MPI_IN_PLACE ? (const unsigned char*)recvbuf + own_offset : sendbuf;
    double exchanged = exchange(context, own_data);

    memcpy(recvbuf, context->data, context->data_bytes);
    atomic_fetch_add_explicit(&context->counters->copied, 1, memory_order_release);
    Allgather_shm_stage_times[2] = MPI_Wtime() - exchanged;
  }
  return MPI_SUCCESS;
}

int Allgather_shm_view(const void* sendbuf,
                       int sendcount,
                       MPI_Datatype datatype,
                       const void** result,
                       MPI_Comm comm,
                       int ranks_per_node)
{
  AllgatherShmContext* context;
  int error;

  if (sendbuf == MPI_IN_PLACE)
  {
    return MPI_ERR_BUFFER;
  }
  if (result == NULL)
  {
    return MPI_ERR_ARG;
  }
  *result = NULL;
  error = bind_context(sendcount, datatype, comm, ranks_per_node, &context);
  if (error != MPI_SUCCESS)
  {
    return error;
  }
  if (context->data_bytes > 0)
  {
    double exchanged = exchange(context, sendbuf);

    context->result_held = 1;
    Allgather_shm_stage_times[2] = MPI_Wtime() - exchanged;
    *result = context->read_only_data;
  }
  return MPI_SUCCESS;
}

void Allgather_shm_release_cached(void)
{
  while (cached_contexts != NULL)
  {
    AllgatherShmContext* context = cached_contexts;
    cached_contexts = context->next;
    destroy_context(context);
  }
}
