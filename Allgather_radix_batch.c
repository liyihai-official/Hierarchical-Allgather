/**
 * @file Allgather_radix_batch.c
 * @brief 分组、分批的 k 进制 Allgather（Allgather_radix_batch）实现。
 *
 * @details
 * 记号：P 为通信子的进程数；B = ranks_per_group，每组 B 个连续 rank；
 * G = P / B 为组数；k = radix。组 g 中本地 rank 为 t 的进程是全局 rank g·B + t。
 * 组 g 的“组块” D_g 是组内 B 个进程的贡献按本地 rank 顺序拼接成的数据块。
 *
 * 一次调用分三个阶段：
 * 1. 组内收集（见 gather_within_group()）：每组以本地 rank g mod B 为根（根随组号轮换），
 *    用 k-nomial 树把组内 B 个贡献收集到根，根上得到 D_g。
 * 2. 组块分发（见 distribute_group_blocks()）：组 g 的根把 D_g 发给其他每个组中本地 rank
 *    同为 g mod B 的进程。组按每 B 个一批处理，第 s 批中本地 rank t 收到 D_(s·B+t)。
 * 3. 组内交换（见 gather_batch() 和 gather_batch_direct()）：对每一批，组内 B 个进程用
 *    每轮最多 k − 1 个伙伴的 Bruck 算法交换各自持有的组块，结束后组内每个进程都有
 *    这一批的全部组块。
 *
 * 只有第 2 阶段在组与组之间通信，每个组从其他每个组各收到一份组块。若进程按节点连续
 * 编号且 B 等于每节点进程数，一组就是一个节点，第 1、3 阶段都在节点内完成。
 *
 * 两条数据路径：
 * - 原生路径（ALLGATHER_RADIX_BATCH_NATIVE_FAST，且所有进程的类型都是稠密的预定义类型，
 *   见 is_dense_type()）：以 MPI_BYTE 直接传输用户数据，第 2、3 阶段直接写入 recvbuf 的
 *   最终位置，不打包、不做额外拷贝。
 * - 可移植路径：每个贡献先用 MPI_Pack 打包成定长“记录”（4 字节长度头 + 打包数据 + 填充），
 *   按记录传输，最后用 MPI_Unpack 写入 recvbuf；支持非连续、resized 等派生类型。
 *
 * 错误处理：准备阶段（参数检查、内存分配、打包）的错误在没有未完成请求的集合点由
 * agree_status() 统一，所有进程返回同一个错误码；通信进行中的意外错误无法安全恢复，
 * require_communication() 直接 MPI_Abort。
 */
#include "Allgather_radix_batch.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/** @brief 内部常量：三个阶段的消息标签，以及可移植路径中记录头的字节数。 */
enum
{
  ALLGATHER_RADIX_BATCH_TAG_GATHER = 0,          ///< 第 1 阶段（组内收集）的消息标签
  ALLGATHER_RADIX_BATCH_TAG_DISTRIBUTE = 1,      ///< 第 2 阶段（组块分发）的消息标签
  ALLGATHER_RADIX_BATCH_TAG_BRUCK = 2,           ///< 第 3 阶段（组内 Bruck 交换）的消息标签
  ALLGATHER_RADIX_BATCH_RECORD_HEADER_BYTES = 4  ///< 记录头字节数：大端 32 位打包长度
};

_Static_assert(CHAR_BIT == 8, "The record length encoding requires 8-bit bytes");
_Static_assert(INT_MAX <= UINT32_MAX, "The record header must hold an MPI int count");

/**
 * @brief 上下文持有的工作区：一块只增不减的字节缓冲区和一个请求数组。
 *
 * 由 Allgather_radix_batch_context_destroy() 释放。
 */
typedef struct
{
  unsigned char* base;    ///< 缓冲区首地址，由 reserve_workspace() 分配或扩大
  size_t capacity_bytes;  ///< base 当前的容量（字节）
  MPI_Request* requests;  ///< 非阻塞通信请求数组，大小见 request_capacity()
  int request_capacity;   ///< requests 的元素个数
} Workspace;

/**
 * @brief 一次调用的缓冲区布局（所有进程相同）。
 *
 * 可移植路径使用定长记录：记录 = 4 字节长度头 + 最多 pack_capacity 字节的 MPI_Pack 数据
 * + 填充，从而保留每个独立打包单元的边界。原生路径中一个记录就是一个进程的原始数据。
 */
typedef struct
{
  int pack_capacity;            ///< 每个记录可容纳的打包字节数（各进程 MPI_Pack_size 的最大值）；原生路径为 0
  size_t record_bytes;          ///< 记录字节数：可移植路径为 4 + pack_capacity，原生路径为每个进程的数据字节数
  size_t group_bytes;           ///< 组块字节数 = B × record_bytes
  size_t assigned_bytes;        ///< 第 2 阶段收到的组块，每批一个：批数 × group_bytes（仅可移植路径）
  size_t batch_bytes;           ///< 第 3 阶段一批的循环缓冲区：min(G, B) × group_bytes（仅可移植路径）
  size_t storage_bytes;         ///< 工作区总需求（字节）
  size_t receive_stride_bytes;  ///< recvbuf 中相邻进程贡献的间距 = recvcount × extent(recvtype)
  int native_bytes;             ///< 非 0 表示原生路径（直接以 MPI_BYTE 传输）
} CallLayout;

/**
 * @brief Allgather_radix_batch_prepare() 绑定的调用参数和布局。
 *
 * 之后 Allgather_radix_batch_ctx() 的参数必须与这里一致（就地模式下忽略 sendcount/sendtype）。
 */
typedef struct
{
  int ready;              ///< 非 0 表示已成功 prepare
  int empty;              ///< 非 0 表示每个贡献都是 0 字节，调用直接返回
  int in_place;           ///< 1 表示使用 MPI_IN_PLACE
  int sendcount;          ///< 绑定的发送个数（就地模式不使用）
  int recvcount;          ///< 绑定的每个进程的接收个数
  MPI_Datatype sendtype;  ///< 绑定的发送类型（就地模式不使用）
  MPI_Datatype recvtype;  ///< 绑定的接收类型
  CallLayout layout;      ///< 对应的缓冲区布局
} PreparedCall;

/**
 * @brief 通信上下文（头文件中的不透明类型）。
 *
 * 由 Allgather_radix_batch_context_create() 创建，拥有一个复制出来的私有通信子和工作区。
 * 组 g 由全局 rank g·B … g·B + B − 1 组成。
 */
struct AllgatherRadixBatchContext
{
  MPI_Comm comm;          ///< 私有通信子（MPI_Comm_dup 得到，错误处理器为 MPI_ERRORS_RETURN）
  int rank;               ///< 本进程在 comm 中的 rank
  int process_count;      ///< 进程总数 P
  int radix;              ///< 基数 k（≥ 2）：第 1 阶段树的分支数，第 3 阶段每轮最多 k − 1 个伙伴
  int ranks_per_group;    ///< 每组进程数 B
  int group_index;        ///< 本进程所在的组号 = rank / B
  int local_rank;         ///< 组内本地 rank = rank mod B
  int group_count;        ///< 组数 G = P / B
  Workspace workspace;    ///< 工作区和请求数组
  PreparedCall prepared;  ///< 最近一次成功 prepare 的绑定
};

/**
 * @brief 工作区内各缓冲区的视图。
 *
 * 这些指针都指向 Workspace.base 内部，不拥有内存。原生路径只用 normalized_group 和
 * ordered_group，其余为 NULL。
 */
typedef struct
{
  unsigned char* local_record;      ///< 本进程打包后的记录（仅可移植路径）
  unsigned char* normalized_group;  ///< 第 1 阶段按“相对根的编号”排列的记录
  unsigned char* ordered_group;     ///< 根上按真实本地 rank 排列的组块 D_g
  unsigned char* assigned_blocks;   ///< 第 2 阶段收到的组块，每批一个槽位（仅可移植路径）
  unsigned char* cyclic_batch;      ///< 第 3 阶段按循环顺序存放一批组块（仅可移植路径）
} BufferViews;

/**
 * @brief 返回两个 int 中较小的一个。
 * @param[in] left  第一个值
 * @param[in] right 第二个值
 * @return left 与 right 中较小的一个。
 */
static int min_int(int left, int right)
{
  return left < right ? left : right;
}

/**
 * @brief 带溢出检查的 size_t 加法。
 * @param[in]  left   加数
 * @param[in]  right  加数
 * @param[out] result 成功时为 left + right
 * @return 成功返回 MPI_SUCCESS；结果超过 SIZE_MAX 时返回 MPI_ERR_COUNT，result 不变。
 */
static int checked_add(size_t left,
                       size_t right,
                       size_t* result)
{
  if (right > SIZE_MAX - left)
  {
    return MPI_ERR_COUNT;
  }
  *result = left + right;
  return MPI_SUCCESS;
}

/**
 * @brief 带溢出检查的 size_t 乘法。
 * @param[in]  left   因数
 * @param[in]  right  因数
 * @param[out] result 成功时为 left × right
 * @return 成功返回 MPI_SUCCESS；结果超过 SIZE_MAX 时返回 MPI_ERR_COUNT，result 不变。
 */
static int checked_multiply(size_t left,
                            size_t right,
                            size_t* result)
{
  if (left != 0 && right > SIZE_MAX / left)
  {
    return MPI_ERR_COUNT;
  }
  *result = left * right;
  return MPI_SUCCESS;
}

/**
 * @brief 检查 instances 个 type 实例覆盖的地址范围能否表示，在 MPI 访问缓冲区之前拒绝不可表示的跨度。
 *
 * 要求 extent 与 true extent 非负；考虑负的 true lower bound 之后，
 * (instances − 1) × extent 加上单个实例的覆盖范围不能超过 PTRDIFF_MAX。
 *
 * @param[in]  type      MPI 数据类型
 * @param[in]  instances 实例个数（为 0 时只检查 extent）
 * @param[out] extent    该类型的 extent（MPI_Type_get_extent 的结果）
 * @return MPI_SUCCESS；类型查询失败时返回 MPI 的错误码；extent 为负或过大时返回 MPI_ERR_TYPE；
 *         跨度溢出时返回 MPI_ERR_COUNT。
 */
static int validate_type_span(MPI_Datatype type,
                              size_t instances,
                              MPI_Aint* extent)
{
  MPI_Aint lower_bound = 0;
  MPI_Aint true_lower_bound = 0;
  MPI_Aint true_extent = 0;
  size_t last_instance;
  size_t positive_end;
  uintmax_t negative_prefix;
  int error = MPI_Type_get_extent(type, &lower_bound, extent);

  if (error != MPI_SUCCESS)
  {
    return error;
  }
  error = MPI_Type_get_true_extent(type, &true_lower_bound, &true_extent);
  if (error != MPI_SUCCESS)
  {
    return error;
  }
  if (*extent < 0 || true_extent < 0 || (uintmax_t)*extent > SIZE_MAX ||
      (uintmax_t)true_extent > SIZE_MAX)
  {
    return MPI_ERR_TYPE;
  }
  if (instances == 0 || true_extent == 0)
  {
    return MPI_SUCCESS;
  }
  error = checked_multiply(instances - 1, (size_t)*extent, &last_instance);
  if (error != MPI_SUCCESS || last_instance > PTRDIFF_MAX)
  {
    return MPI_ERR_COUNT;
  }
  if (true_lower_bound < 0)
  {
    negative_prefix = (uintmax_t)(-(true_lower_bound + 1)) + 1;
    if (negative_prefix > (uintmax_t)PTRDIFF_MAX + 1)
    {
      return MPI_ERR_COUNT;
    }
    positive_end = (uintmax_t)true_extent > negative_prefix
                     ? (size_t)((uintmax_t)true_extent - negative_prefix)
                     : 0;
  }
  else
  {
    if ((uintmax_t)true_lower_bound > SIZE_MAX)
    {
      return MPI_ERR_COUNT;
    }
    error = checked_add((size_t)true_lower_bound, (size_t)true_extent, &positive_end);
    if (error != MPI_SUCCESS)
    {
      return error;
    }
  }
  error = checked_add(last_instance, positive_end, &positive_end);
  return error == MPI_SUCCESS && positive_end <= PTRDIFF_MAX ? MPI_SUCCESS : MPI_ERR_COUNT;
}

/**
 * @brief 检查内部通信调用的返回值，失败即中止。
 *
 * 通信进行到一半出错时各进程的状态已不一致，无法再安全地进入下一次错误协商，
 * 因此直接 MPI_Abort(comm)；若 MPI_Abort 返回，再调用 abort()。
 *
 * @param[in] comm  中止所用的通信子
 * @param[in] error MPI 调用的返回值；为 MPI_SUCCESS 时什么都不做
 */
static void require_communication(MPI_Comm comm, int error)
{
  if (error != MPI_SUCCESS)
  {
    (void)MPI_Abort(comm, error);
    abort();
  }
}

/**
 * @brief 集合地协商错误，使所有进程返回同一个结果。
 *
 * 必须在没有未完成请求的集合点由所有进程调用。先用 MPI_Allreduce(MPI_MIN) 找出出错的
 * 最小 rank，再由它广播自己的错误码。
 *
 * @param[in] context     上下文（使用其 comm、rank 和 process_count）
 * @param[in] local_error 本进程的错误码
 * @return 所有进程都成功时为 MPI_SUCCESS，否则为出错的最小 rank 上的错误码。
 */
static int agree_status(const AllgatherRadixBatchContext* context, int local_error)
{
  int first_error_rank = context->process_count;
  int candidate = local_error == MPI_SUCCESS ? context->process_count : context->rank;

  require_communication(
    context->comm,
    MPI_Allreduce(&candidate,
                  &first_error_rank,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  context->comm));

  if (first_error_rank == context->process_count)
  {
    return MPI_SUCCESS;
  }
  require_communication(
    context->comm,
    MPI_Bcast(&local_error,
              1,
              MPI_INT,
              first_error_rank,
              context->comm));
  return local_error;
}

/**
 * @brief 计算 (value + offset) mod modulus，不会发生 int 溢出。
 * @param[in] value   被加数
 * @param[in] offset  加数
 * @param[in] modulus 模数（组大小或本批组数）
 * @return (value + offset) mod modulus。
 * @pre 0 ≤ value < modulus，0 ≤ offset < modulus。
 */
static int add_modulo(int value, int offset, int modulus)
{
  /* value and offset are in [0, modulus), without value + offset overflow. */
  return value >= modulus - offset ? value - (modulus - offset) : value + offset;
}

/**
 * @brief 计算 (value − offset) mod modulus，结果在 [0, modulus) 内。
 * @param[in] value   被减数
 * @param[in] offset  减数
 * @param[in] modulus 模数（组大小）
 * @return (value − offset) mod modulus。
 * @pre 0 ≤ value < modulus，0 ≤ offset < modulus。
 */
static int subtract_modulo(int value, int offset, int modulus)
{
  return value >= offset ? value - offset : modulus - (offset - value);
}

/**
 * @brief 下一轮的距离：min(distance × radix, group_size)，计算中不会溢出。
 *
 * 第 1、3 阶段每轮把距离乘以 k；最后一轮截到组大小。
 * @param[in] distance   本轮的距离（≥ 1）
 * @param[in] radix      基数 k
 * @param[in] group_size 组大小 B
 * @return 下一轮的距离；返回 group_size 表示已覆盖整组。
 */
static int next_distance(int distance, int radix, int group_size)
{
  return distance > (group_size - 1) / radix ? group_size : distance * radix;
}

/**
 * @brief 组 group_index 的根（本地 rank）：group_index mod B。
 *
 * 根随组号轮换，所以第 2 阶段第 s 批中组 s·B + t 的根正好是本地 rank t，跨组发送分散到
 * 组内不同的进程上。第 1、2 阶段必须使用同一个根。
 * @param[in] context     上下文（使用 ranks_per_group）
 * @param[in] group_index 组号
 * @return 根的本地 rank。
 */
static int root_local_rank(const AllgatherRadixBatchContext* context, int group_index)
{
  return group_index % context->ranks_per_group;
}

/**
 * @brief 组 group_index 中本地 rank 为 local_rank 的进程的全局 rank：group_index × B + local_rank。
 * @param[in] context     上下文（使用 ranks_per_group）
 * @param[in] group_index 组号（< G，所以结果不会溢出）
 * @param[in] local_rank  组内本地 rank
 * @return 全局 rank。
 */
static int global_rank(const AllgatherRadixBatchContext* context, int group_index, int local_rank)
{
  /* Group indices are bounded by process_count / ranks_per_group. */
  return group_index * context->ranks_per_group + local_rank;
}

/**
 * @brief 计算一次调用中最多同时挂起的请求数。
 *
 * 取以下两者的最大值（至少为 1）：
 * - 第 3 阶段：每轮最多 min(k − 1, B − 1) 个伙伴，每个伙伴最多收、发各 2 条消息
 *   （循环区间在末尾折返时拆成两段），共 4 × 伙伴数；
 * - 第 2 阶段：根向其余 G − 1 个组各发一条消息。
 * 第 1 阶段每轮最多 k − 1 个接收，不超过上面的值。
 *
 * @param[in]  context  已设置 radix、ranks_per_group 和 group_count 的上下文
 * @param[out] capacity 请求数组的大小
 * @return MPI_SUCCESS；4 × 伙伴数超出 int 时返回 MPI_ERR_COUNT。
 */
static int request_capacity(const AllgatherRadixBatchContext* context, int* capacity)
{
  int partners = min_int(context->radix - 1, context->ranks_per_group - 1);
  if (partners > INT_MAX / 4)
  {
    return MPI_ERR_COUNT;
  }
  *capacity = 4 * partners;
  if (*capacity < context->group_count - 1)
  {
    *capacity = context->group_count - 1;
  }
  if (*capacity == 0)
  {
    *capacity = 1;
  }
  return MPI_SUCCESS;
}

/**
 * @brief 集合地创建一个独立的通信上下文。
 *
 * 复制 comm 得到私有通信子（错误处理器设为 MPI_ERRORS_RETURN），检查所有进程传入相同的
 * radix 和 ranks_per_group，并分配请求数组。组由连续的 rank 组成，不推断物理拓扑。
 *
 * @param[in]  comm            有效的组内通信子（不支持组间通信子）
 * @param[in]  radix           基数 k，≥ 2，所有进程相同
 * @param[in]  ranks_per_group 每组进程数 B，≥ 1 且整除通信子大小，所有进程相同
 * @param[out] output          成功时指向新上下文；失败时置为 NULL（指针本身非 NULL 时）
 * @return MPI_SUCCESS；comm 为 MPI_COMM_NULL 或组间通信子时返回 MPI_ERR_COMM（此时不做集合通信）；
 *         参数非法或各进程不一致时返回 MPI_ERR_ARG；B 不整除进程数时返回 MPI_ERR_DIMS；
 *         内存不足时返回 MPI_ERR_NO_MEM。可恢复的错误在所有进程上返回相同的错误码。
 * @note 集合操作，comm 的所有进程都必须调用。
 */
int Allgather_radix_batch_context_create(MPI_Comm comm,
                                         int radix,
                                         int ranks_per_group,
                                         AllgatherRadixBatchContext** output)
{
  AllgatherRadixBatchContext setup = {0};
  AllgatherRadixBatchContext* created = NULL;
  int local_error = MPI_SUCCESS;
  int error;
  int is_intercommunicator = 0;
  int configuration[2] = {radix, ranks_per_group};
  int minimum[2];
  int maximum[2];
  int capacity = 0;
  size_t request_bytes = 0;

  if (output != NULL)
  {
    *output = NULL;
  }
  if (comm == MPI_COMM_NULL)
  {
    return MPI_ERR_COMM;
  }
  require_communication(comm, MPI_Comm_test_inter(comm, &is_intercommunicator));
  if (is_intercommunicator)
  {
    return MPI_ERR_COMM;
  }

  require_communication(comm, MPI_Comm_dup(comm, &setup.comm));
  require_communication(setup.comm, MPI_Comm_set_errhandler(setup.comm, MPI_ERRORS_RETURN));
  require_communication(setup.comm, MPI_Comm_rank(setup.comm, &setup.rank));
  require_communication(setup.comm, MPI_Comm_size(setup.comm, &setup.process_count));

  require_communication(
    setup.comm,
    MPI_Allreduce(configuration,
                  minimum,
                  2,
                  MPI_INT,
                  MPI_MIN,
                  setup.comm));

  require_communication(
    setup.comm,
    MPI_Allreduce(configuration,
                  maximum,
                  2,
                  MPI_INT,
                  MPI_MAX,
                  setup.comm));

  if (output == NULL || radix < 2 || ranks_per_group < 1 || minimum[0] != maximum[0] ||
      minimum[1] != maximum[1])
  {
    local_error = MPI_ERR_ARG;
  }
  else if (ranks_per_group > setup.process_count ||
           setup.process_count % ranks_per_group != 0)
  {
    local_error = MPI_ERR_DIMS;
  }
  error = agree_status(&setup, local_error);
  if (error != MPI_SUCCESS)
  {
    goto cleanup;
  }

  setup.radix = radix;
  setup.ranks_per_group = ranks_per_group;
  setup.group_index = setup.rank / ranks_per_group;
  setup.local_rank = setup.rank % ranks_per_group;
  setup.group_count = setup.process_count / ranks_per_group;

  local_error = request_capacity(&setup, &capacity);
  if (local_error == MPI_SUCCESS)
  {
    local_error = checked_multiply((size_t)capacity, sizeof(MPI_Request), &request_bytes);
  }
  if (local_error == MPI_SUCCESS)
  {
    created = calloc(1, sizeof(*created));
    if (created == NULL)
    {
      local_error = MPI_ERR_NO_MEM;
    }
    else
    {
      *created = setup;
      created->workspace.requests = malloc(request_bytes);
      created->workspace.request_capacity = capacity;
      if (created->workspace.requests == NULL)
      {
        local_error = MPI_ERR_NO_MEM;
      }
    }
  }
  error = agree_status(&setup, local_error);
  if (error == MPI_SUCCESS)
  {
    *output = created;
    return MPI_SUCCESS;
  }

cleanup:
  if (created != NULL)
  {
    free(created->workspace.requests);
    free(created);
  }
  require_communication(setup.comm, MPI_Comm_free(&setup.comm));
  return error;
}

/**
 * @brief 检查一次调用的本地参数并计算布局所需的量，不访问用户数据。
 *
 * 检查个数、类型和缓冲区指针，要求发送字节数等于接收字节数且不超过 INT_MAX，并检查
 * recvbuf 中 P 个接收块的地址范围可以表示。就地模式（sendbuf == MPI_IN_PLACE）下用
 * recvcount/recvtype 代替 sendcount/sendtype。
 *
 * @param[in]  context        上下文
 * @param[in]  sendbuf        发送缓冲区或 MPI_IN_PLACE（只比较地址）
 * @param[in]  sendcount      发送个数
 * @param[in]  sendtype       发送类型
 * @param[in]  recvbuf        接收缓冲区（只比较地址）
 * @param[in]  recvcount      每个进程的接收个数
 * @param[in]  recvtype       接收类型
 * @param[out] pack_bound     本进程贡献的 MPI_Pack_size 上界（数据为空时为 0）
 * @param[out] payload_bytes  本进程贡献的字节数（类型大小 × 个数）
 * @param[out] receive_stride recvbuf 中相邻贡献的间距（字节；数据为空时为 0）
 * @return MPI_SUCCESS，或 MPI_ERR_COUNT / MPI_ERR_TYPE / MPI_ERR_TRUNCATE / MPI_ERR_BUFFER，
 *         或 MPI 类型查询返回的错误码。
 * @note 只做本地检查，调用者需要用 agree_status() 统一结果。
 */
static int validate_call(const AllgatherRadixBatchContext* context,
                         const void* sendbuf,
                         int sendcount,
                         MPI_Datatype sendtype,
                         void* recvbuf,
                         int recvcount,
                         MPI_Datatype recvtype,
                         int* pack_bound,
                         unsigned long long* payload_bytes,
                         size_t* receive_stride)
{
  int send_type_bytes = 0;
  int receive_type_bytes = 0;
  int effective_count;
  MPI_Datatype effective_type;
  MPI_Aint extent = 0;
  size_t send_bytes = 0;
  size_t receive_bytes = 0;
  size_t last_block_offset = 0;
  size_t receive_instances = 0;
  int error;

  if (recvcount < 0 || (sendbuf != MPI_IN_PLACE && sendcount < 0))
  {
    return MPI_ERR_COUNT;
  }
  if (recvtype == MPI_DATATYPE_NULL ||
      (sendbuf != MPI_IN_PLACE && sendtype == MPI_DATATYPE_NULL))
  {
    return MPI_ERR_TYPE;
  }
  effective_count = sendbuf == MPI_IN_PLACE ? recvcount : sendcount;
  effective_type = sendbuf == MPI_IN_PLACE ? recvtype : sendtype;

  error = MPI_Type_size(effective_type, &send_type_bytes);
  if (error != MPI_SUCCESS)
  {
    return error;
  }
  error = MPI_Type_size(recvtype, &receive_type_bytes);
  if (error != MPI_SUCCESS)
  {
    return error;
  }
  if (send_type_bytes < 0 || receive_type_bytes < 0)
  {
    return MPI_ERR_COUNT;
  }
  error = checked_multiply((size_t)effective_count, (size_t)send_type_bytes, &send_bytes);
  if (error != MPI_SUCCESS)
  {
    return error;
  }
  error = checked_multiply((size_t)recvcount, (size_t)receive_type_bytes, &receive_bytes);
  if (error != MPI_SUCCESS || send_bytes > INT_MAX || receive_bytes > INT_MAX)
  {
    return MPI_ERR_COUNT;
  }
  if (send_bytes != receive_bytes)
  {
    return MPI_ERR_TRUNCATE;
  }
  if (recvbuf == MPI_IN_PLACE ||
      (receive_bytes > 0 && (recvbuf == NULL || recvbuf == MPI_BOTTOM)) ||
      (send_bytes > 0 && sendbuf != MPI_IN_PLACE && (sendbuf == NULL || sendbuf == MPI_BOTTOM)))
  {
    return MPI_ERR_BUFFER;
  }

  error = checked_multiply((size_t)context->process_count, (size_t)recvcount, &receive_instances);
  if (error != MPI_SUCCESS)
  {
    return error;
  }
  error = validate_type_span(recvtype, receive_instances, &extent);
  if (error != MPI_SUCCESS)
  {
    return error;
  }
  *receive_stride = 0;
  *pack_bound = 0;
  *payload_bytes = (unsigned long long)send_bytes;
  if (send_bytes == 0)
  {
    return MPI_SUCCESS;
  }
  error = checked_multiply((size_t)recvcount, (size_t)extent, receive_stride);
  if (error != MPI_SUCCESS)
  {
    return error;
  }
  error =
    checked_multiply((size_t)(context->process_count - 1), *receive_stride, &last_block_offset);
  if (error != MPI_SUCCESS || last_block_offset > PTRDIFF_MAX)
  {
    return MPI_ERR_COUNT;
  }
  if (sendbuf != MPI_IN_PLACE)
  {
    MPI_Aint send_extent = 0;
    error = validate_type_span(sendtype, (size_t)sendcount, &send_extent);
    if (error != MPI_SUCCESS)
    {
      return error;
    }
  }
  error = MPI_Pack_size(effective_count, effective_type, context->comm, pack_bound);
  if (error != MPI_SUCCESS)
  {
    return error;
  }
  return *pack_bound < 0 ? MPI_ERR_COUNT : MPI_SUCCESS;
}

/**
 * @brief 计算可移植路径（打包记录）的布局。
 *
 * 记录 = 4 字节长度头 + pack_capacity 字节；组块 = B 个记录；批数 = ⌈G / B⌉。
 * 工作区依次为：本地记录、normalized_group、ordered_group、assigned_blocks（每批一个组块）、
 * cyclic_batch（min(G, B) 个组块）。
 *
 * @param[in]  context        上下文
 * @param[in]  pack_capacity  每个记录的打包容量（所有进程 pack_bound 的最大值）
 * @param[in]  receive_stride recvbuf 中相邻贡献的间距（字节）
 * @param[out] layout         计算出的布局
 * @return MPI_SUCCESS；单条消息（记录、组块、一批组块）超过 INT_MAX 或总量溢出时返回 MPI_ERR_COUNT。
 */
static int make_call_layout(const AllgatherRadixBatchContext* context,
                            int pack_capacity,
                            size_t receive_stride,
                            CallLayout* layout)
{
  int batch_count = context->group_count / context->ranks_per_group;
  int maximum_batch_groups = min_int(context->group_count, context->ranks_per_group);
  size_t total;
  int error;

  if (context->group_count % context->ranks_per_group != 0)
  {
    ++batch_count;
  }
  memset(layout, 0, sizeof(*layout));
  layout->pack_capacity = pack_capacity;
  layout->receive_stride_bytes = receive_stride;
  error = checked_add((size_t)pack_capacity, ALLGATHER_RADIX_BATCH_RECORD_HEADER_BYTES, &layout->record_bytes);
  if (error != MPI_SUCCESS || layout->record_bytes > INT_MAX)
  {
    return MPI_ERR_COUNT;
  }
  error = checked_multiply((size_t)context->ranks_per_group, layout->record_bytes, &layout->group_bytes);
  if (error != MPI_SUCCESS || layout->group_bytes > INT_MAX)
  {
    return MPI_ERR_COUNT;
  }
  error = checked_multiply((size_t)batch_count, layout->group_bytes, &layout->assigned_bytes);
  if (error != MPI_SUCCESS)
  {
    return error;
  }
  error =
    checked_multiply((size_t)maximum_batch_groups, layout->group_bytes, &layout->batch_bytes);
  if (error != MPI_SUCCESS || layout->batch_bytes > INT_MAX)
  {
    return MPI_ERR_COUNT;
  }

  total = layout->record_bytes;
  error = checked_add(total, layout->group_bytes, &total);
  if (error == MPI_SUCCESS)
  {
    error = checked_add(total, layout->group_bytes, &total);
  }
  if (error == MPI_SUCCESS)
  {
    error = checked_add(total, layout->assigned_bytes, &total);
  }
  if (error == MPI_SUCCESS)
  {
    error = checked_add(total, layout->batch_bytes, &total);
  }
  layout->storage_bytes = total;
  return error;
}

/**
 * @brief 集合地确保工作区至少有 required_bytes 字节，以事务方式扩大。
 *
 * 先分配新缓冲区，再用 agree_status() 协商：只要有一个进程分配失败，所有进程都保留旧的
 * 工作区并返回同一个错误。工作区只增不减，扩大时不保留旧内容。
 *
 * @param[in,out] context        上下文
 * @param[in]     required_bytes 需要的字节数
 * @return MPI_SUCCESS 或 MPI_ERR_NO_MEM（所有进程一致）。
 * @note 集合操作。
 */
static int reserve_workspace(AllgatherRadixBatchContext* context, size_t required_bytes)
{
  unsigned char* replacement = NULL;
  int local_error = MPI_SUCCESS;
  int error;

  if (required_bytes > context->workspace.capacity_bytes)
  {
    replacement = malloc(required_bytes);
    if (replacement == NULL)
    {
      local_error = MPI_ERR_NO_MEM;
    }
  }
  error = agree_status(context, local_error);
  if (error != MPI_SUCCESS)
  {
    free(replacement);
    return error;
  }
  if (replacement != NULL)
  {
    free(context->workspace.base);
    context->workspace.base = replacement;
    context->workspace.capacity_bytes = required_bytes;
  }
  return MPI_SUCCESS;
}

/**
 * @brief 按布局把工作区划分为各个缓冲区视图。
 * @param[in] context 上下文（使用 workspace.base）
 * @param[in] layout  当前布局
 * @return 各缓冲区在工作区中的起始地址；原生路径只有 normalized_group 和 ordered_group，其余为 NULL。
 */
static BufferViews buffer_views(AllgatherRadixBatchContext* context, const CallLayout* layout)
{
  BufferViews views;
  if (layout->native_bytes)
  {
    views.local_record = NULL;
    views.normalized_group = context->workspace.base;
    views.ordered_group = views.normalized_group + layout->group_bytes;
    views.assigned_blocks = NULL;
    views.cyclic_batch = NULL;
    return views;
  }
  views.local_record = context->workspace.base;
  views.normalized_group = views.local_record + layout->record_bytes;
  views.ordered_group = views.normalized_group + layout->group_bytes;
  views.assigned_blocks = views.ordered_group + layout->group_bytes;
  views.cyclic_batch = views.assigned_blocks + layout->assigned_bytes;
  return views;
}

/**
 * @brief 把记录中打包数据的长度以大端 4 字节写入记录头。
 * @param[out] header 记录头（4 字节）
 * @param[in]  length 打包字节数（0 … INT_MAX）
 */
static void encode_length(unsigned char* header, int length)
{
  uint32_t value = (uint32_t)length;
  header[0] = (unsigned char)(value >> 24);
  header[1] = (unsigned char)(value >> 16);
  header[2] = (unsigned char)(value >> 8);
  header[3] = (unsigned char)value;
}

/**
 * @brief 从大端 4 字节记录头读出打包数据的长度。
 * @param[in] header 记录头（4 字节）
 * @return 打包字节数。
 */
static uint32_t decode_length(const unsigned char* header)
{
  return ((uint32_t)header[0] << 24) | ((uint32_t)header[1] << 16) | ((uint32_t)header[2] << 8) |
         (uint32_t)header[3];
}

/**
 * @brief 把本进程的贡献打包成一个定长记录（可移植路径）。
 *
 * 先把整个记录清零（填充部分也会被发送，不能暴露工作区中未初始化的内容），再把数据
 * MPI_Pack 到记录头之后，最后写入实际的打包长度。就地模式下数据取自 recvbuf 中本进程的接收块。
 *
 * @param[in]  context   上下文
 * @param[in]  layout    当前布局
 * @param[out] record    目标记录（record_bytes 字节）
 * @param[in]  sendbuf   发送缓冲区或 MPI_IN_PLACE
 * @param[in]  sendcount 发送个数
 * @param[in]  sendtype  发送类型
 * @param[in]  recvbuf   接收缓冲区（就地模式下的数据来源）
 * @param[in]  recvcount 每个进程的接收个数（就地模式下使用）
 * @param[in]  recvtype  接收类型（就地模式下使用）
 * @return MPI_SUCCESS；MPI_Pack 的错误码；打包长度超过容量时返回 MPI_ERR_TRUNCATE。
 */
static int pack_local_record(const AllgatherRadixBatchContext* context,
                             const CallLayout* layout,
                             unsigned char* record,
                             const void* sendbuf,
                             int sendcount,
                             MPI_Datatype sendtype,
                             void* recvbuf,
                             int recvcount,
                             MPI_Datatype recvtype)
{
  const void* source = sendbuf;
  int position = 0;
  int error;

  if (sendbuf == MPI_IN_PLACE)
  {
    source = (unsigned char*)recvbuf + (size_t)context->rank * layout->receive_stride_bytes;
    sendcount = recvcount;
    sendtype = recvtype;
  }
  /* Padding is sent too: never expose uninitialized workspace contents. */
  memset(record, 0, layout->record_bytes);
  error =
    MPI_Pack(source, sendcount, sendtype, record + ALLGATHER_RADIX_BATCH_RECORD_HEADER_BYTES, layout->pack_capacity, &position, context->comm);
  if (error != MPI_SUCCESS)
  {
    return error;
  }
  if (position < 0 || position > layout->pack_capacity)
  {
    return MPI_ERR_TRUNCATE;
  }
  encode_length(record, position);
  return MPI_SUCCESS;
}

/**
 * @brief 第 1 阶段：用 k-nomial 树把组内 B 个记录收集到本组的根。
 *
 * 根为本地 rank r = group_index mod B（见 root_local_rank()）。各进程使用相对根的编号
 * n = (local_rank − r) mod B，记录按 n 的顺序存放在 normalized_group 中。距离为 d 的一轮里，
 * 跨度为 d·k 的子树的根接收位于 d、2d、…、(k − 1)d 处的子节点已收集好的连续记录（每个最多
 * d 个）；其他进程把自己收集到的记录发给子树的根后退出。最后根把记录从相对编号重排为真实
 * 本地 rank 顺序，写入 ordered_group，得到组块 D_g。
 *
 * 约定：返回时只有根的 ordered_group 有效；子树根在内部使用相对编号。
 *
 * @param[in,out] context      上下文（使用其请求数组）
 * @param[in]     layout       当前布局（record_bytes）
 * @param[in,out] views        工作区视图（normalized_group、ordered_group）
 * @param[in]     local_record 本进程的记录：可移植路径为打包记录，原生路径为用户数据
 *                             （sendbuf，或就地模式下 recvbuf 中本进程的块）
 * @note 通信失败时中止（见 require_communication()）。
 */
static void gather_within_group(AllgatherRadixBatchContext* context,
                                const CallLayout* layout,
                                BufferViews* views,
                                const void* local_record)
{
  int group_size = context->ranks_per_group;
  int root = root_local_rank(context, context->group_index);
  int normalized_rank = subtract_modulo(context->local_rank, root, group_size);
  int distance = 1;

  memcpy(views->normalized_group + (size_t)normalized_rank * layout->record_bytes,
         local_record,
         layout->record_bytes);

  while (distance < group_size)
  {
    int span = next_distance(distance, context->radix, group_size);
    /* A capped final span covers the entire group. Earlier spans are exact. */
    int subtree_start = span == group_size ? 0 : (normalized_rank / span) * span;
    int offset = normalized_rank - subtree_start;
    int request_count = 0;

    if (offset == 0)
    {
      for (int child_offset = distance; child_offset < group_size - subtree_start;)
      {
        int child_normalized = subtree_start + child_offset;
        int child_local = add_modulo(child_normalized, root, group_size);
        int child_rank = global_rank(context, context->group_index, child_local);
        int records = min_int(distance, group_size - child_normalized);
        int bytes = (int)((size_t)records * layout->record_bytes);

        require_communication(context->comm,
                              MPI_Irecv(views->normalized_group +
                                          (size_t)child_normalized * layout->record_bytes,
                                        bytes,
                                        MPI_BYTE,
                                        child_rank,
                                        ALLGATHER_RADIX_BATCH_TAG_GATHER,
                                        context->comm,
                                        &context->workspace.requests[request_count++]));
        if (child_offset > span - 1 - distance)
        {
          break;
        }
        child_offset += distance;
      }
      require_communication(
        context->comm,
        MPI_Waitall(request_count, context->workspace.requests, MPI_STATUSES_IGNORE));
    }
    else
    {
      int parent_local = add_modulo(subtree_start, root, group_size);
      int parent_rank = global_rank(context, context->group_index, parent_local);
      int records = min_int(distance, group_size - normalized_rank);
      int bytes = (int)((size_t)records * layout->record_bytes);
      MPI_Request request;

      require_communication(
        context->comm,
        MPI_Isend(views->normalized_group + (size_t)normalized_rank * layout->record_bytes,
                  bytes,
                  MPI_BYTE,
                  parent_rank,
                  ALLGATHER_RADIX_BATCH_TAG_GATHER,
                  context->comm,
                  &request));
      require_communication(context->comm, MPI_Wait(&request, MPI_STATUS_IGNORE));
      break;
    }
    distance = span;
  }

  if (context->local_rank == root)
  {
    for (int local_rank = 0; local_rank < group_size; ++local_rank)
    {
      int normalized = subtract_modulo(local_rank, root, group_size);
      memcpy(views->ordered_group + (size_t)local_rank * layout->record_bytes,
             views->normalized_group + (size_t)normalized * layout->record_bytes,
             layout->record_bytes);
    }
  }
}

/**
 * @brief 第 2 阶段：把每个组块 D_g 分发到其他各组，组按每 B 个一批处理。
 *
 * 第 s 批包含组 s·B … s·B + m − 1，其中 m = min(B, G − s·B)。本地 rank t < m 的进程负责
 * 组 g = s·B + t：组 g 的根就是组 g 中本地 rank 为 g mod B = t 的进程（与第 1 阶段选的根
 * 相同），它把 D_g 放进自己的槽位，并发送给其他每个组中本地 rank 为 t 的进程；其他组中
 * 本地 rank 为 t 的进程从这个根接收 D_g。因此本地 rank t 在第 s 个槽位得到 D_(s·B+t)，
 * 有效槽位数为 G / B + (t < G mod B)。一批的所有请求完成后才处理下一批。
 *
 * 槽位：原生路径是 recvbuf 中组 g 的最终位置（receive_base + g × group_bytes）；
 * 可移植路径是 assigned_blocks 中的第 s 个组块。
 *
 * @param[in,out] context      上下文
 * @param[in]     layout       当前布局
 * @param[in,out] views        工作区视图（根读取 ordered_group；可移植路径写入 assigned_blocks）
 * @param[out]    receive_base 用户的 recvbuf（仅原生路径使用）
 */
static void distribute_group_blocks(AllgatherRadixBatchContext* context,
                                    const CallLayout* layout,
                                    BufferViews* views,
                                    unsigned char* receive_base)
{
  int batch_begin = 0;
  int batch_index = 0;

  while (batch_begin < context->group_count)
  {
    int batch_groups = min_int(context->ranks_per_group, context->group_count - batch_begin);
    int request_count = 0;

    if (context->local_rank < batch_groups)
    {
      int source_group = batch_begin + context->local_rank;
      int source_rank =
        global_rank(context, source_group, root_local_rank(context, source_group));
      unsigned char* slot =
        layout->native_bytes
          ? receive_base + (size_t)source_group * layout->group_bytes
          : views->assigned_blocks + (size_t)batch_index * layout->group_bytes;

      if (context->rank == source_rank)
      {
        memcpy(slot, views->ordered_group, layout->group_bytes);
        for (int destination_group = 0; destination_group < context->group_count;
             ++destination_group)
        {
          if (destination_group != source_group)
          {
            int destination =
              global_rank(context, destination_group, context->local_rank);
            require_communication(
              context->comm,
              MPI_Isend(views->ordered_group,
                        (int)layout->group_bytes,
                        MPI_BYTE,
                        destination,
                        ALLGATHER_RADIX_BATCH_TAG_DISTRIBUTE,
                        context->comm,
                        &context->workspace.requests[request_count++]));
          }
        }
      }
      else
      {
        require_communication(
          context->comm,
          MPI_Irecv(slot,
                    (int)layout->group_bytes,
                    MPI_BYTE,
                    source_rank,
                    ALLGATHER_RADIX_BATCH_TAG_DISTRIBUTE,
                    context->comm,
                    &context->workspace.requests[request_count++]));
      }
    }
    require_communication(
      context->comm,
      MPI_Waitall(request_count,
                  context->workspace.requests,
                  MPI_STATUSES_IGNORE));
    batch_begin += batch_groups;
    ++batch_index;
  }
}

/**
 * @brief 统计循环区间中有效组块的个数。
 *
 * 区间从 start 开始、长度为 length（≤ group_size），按 group_size 取模折返；只统计落在
 * [0, valid_groups) 内的位置。最后一批不满 B 个组时 valid_groups < B。
 * @param[in] start        区间起点（0 ≤ start < group_size）
 * @param[in] length       区间长度（≤ group_size）
 * @param[in] group_size   模数 B
 * @param[in] valid_groups 本批的组数
 * @return 区间内有效位置的个数。
 */
static int valid_block_count(int start, int length, int group_size, int valid_groups)
{
  int first_length = min_int(length, group_size - start);
  int first_valid = start < valid_groups ? min_int(first_length, valid_groups - start) : 0;
  return first_valid + min_int(length - first_length, valid_groups);
}

/**
 * @brief 第 3 阶段（可移植路径）：组内 B 个进程用 Bruck 算法交换一批组块。
 *
 * 本地 rank t 先把第 2 阶段在本批收到的组块放在 cyclic_batch 开头（若 t < valid_groups）。
 * 距离为 d 的一轮里，与偏移 o = d、2d、…、(k − 1)d（o < B）的伙伴通信：从 t + o 接收区间
 * [t + o, t + o + min(d, B − o)) 内的组块，向 t − o 发送自己区间 [t, t + min(d, B − o)) 内的
 * 组块（均按 B 取模）；然后距离乘以 k，共 ⌈log_k B⌉ 轮。
 *
 * 约定（满批和最后的不满批都适用）：距离为 d 的一轮开始前，rank t 按循环顺序紧密存放
 * [t, t + d)（mod B）内的有效组块。发送只读取这段已有的前缀，接收依次追加后面的区间，
 * 所以接收位置不会与正在发送的数据重叠。结束后 cyclic_batch 按循环顺序存放本批全部
 * valid_groups 个组块，从本进程自己的组块开始（t ≥ valid_groups 时从第 0 个开始）。
 *
 * @param[in,out] context        上下文
 * @param[in]     layout         当前布局（group_bytes）
 * @param[in]     assigned_block 第 2 阶段在本批收到的组块
 * @param[in]     valid_groups   本批的组数（≤ B）
 * @param[out]    cyclic_batch   本批的循环缓冲区（batch_bytes 字节）
 */
static void gather_batch(AllgatherRadixBatchContext* context,
                         const CallLayout* layout,
                         const unsigned char* assigned_block,
                         int valid_groups,
                         unsigned char* cyclic_batch)
{
  int group_size = context->ranks_per_group;
  int local_rank = context->local_rank;
  int distance = 1;

  if (local_rank < valid_groups)
  {
    memcpy(cyclic_batch, assigned_block, layout->group_bytes);
  }
  while (distance < group_size)
  {
    int received_blocks = valid_block_count(local_rank, distance, group_size, valid_groups);
    int request_count = 0;
    int offset = distance;
    int partner_index = 1;

    while (partner_index < context->radix && offset < group_size)
    {
      int interval_length = min_int(distance, group_size - offset);
      int source_local = add_modulo(local_rank, offset, group_size);
      int destination_local = subtract_modulo(local_rank, offset, group_size);
      int receive_blocks =
        valid_block_count(source_local, interval_length, group_size, valid_groups);
      int send_blocks =
        valid_block_count(local_rank, interval_length, group_size, valid_groups);

      if (receive_blocks > 0)
      {
        int source = global_rank(context, context->group_index, source_local);
        require_communication(
          context->comm,
          MPI_Irecv(cyclic_batch + (size_t)received_blocks * layout->group_bytes,
                    (int)((size_t)receive_blocks * layout->group_bytes),
                    MPI_BYTE,
                    source,
                    ALLGATHER_RADIX_BATCH_TAG_BRUCK,
                    context->comm,
                    &context->workspace.requests[request_count++]));
        received_blocks += receive_blocks;
      }
      if (send_blocks > 0)
      {
        int destination = global_rank(context, context->group_index, destination_local);
        require_communication(
          context->comm,
          MPI_Isend(cyclic_batch,
                    (int)((size_t)send_blocks * layout->group_bytes),
                    MPI_BYTE,
                    destination,
                    ALLGATHER_RADIX_BATCH_TAG_BRUCK,
                    context->comm,
                    &context->workspace.requests[request_count++]));
      }
      if (offset > group_size - 1 - distance)
      {
        break;
      }
      offset += distance;
      ++partner_index;
    }
    require_communication(
      context->comm,
      MPI_Waitall(request_count,
                  context->workspace.requests,
                  MPI_STATUSES_IGNORE));
    distance = next_distance(distance, context->radix, group_size);
  }
}

/**
 * @brief 把一批循环排列的组块直接解包到 recvbuf 的最终位置，不需要旋转缓冲区。
 *
 * cyclic_batch 的第 j 个组块属于组 batch_begin + (f + j) mod valid_groups，其中
 * f = local_rank（local_rank < valid_groups 时），否则 f = 0；组块中第 t 个记录属于该组的
 * 全局 rank（组号 × B + t），用 MPI_Unpack 写入 recvbuf 中这个 rank 的接收块。
 *
 * @param[in]  context      上下文
 * @param[in]  layout       当前布局
 * @param[in]  cyclic_batch gather_batch() 的结果
 * @param[in]  batch_begin  本批第一个组的组号
 * @param[in]  valid_groups 本批的组数
 * @param[out] recvbuf      用户的接收缓冲区
 * @param[in]  recvcount    每个进程的接收个数
 * @param[in]  recvtype     接收类型
 * @return MPI_SUCCESS；MPI_Unpack 的错误码；记录长度超过容量或解包长度不符时返回 MPI_ERR_TRUNCATE。
 */
static int unpack_batch(const AllgatherRadixBatchContext* context,
                        const CallLayout* layout,
                        const unsigned char* cyclic_batch,
                        int batch_begin,
                        int valid_groups,
                        void* recvbuf,
                        int recvcount,
                        MPI_Datatype recvtype)
{
  int first_group = context->local_rank < valid_groups ? context->local_rank : 0;

  for (int block = 0; block < valid_groups; ++block)
  {
    int source_group = batch_begin + add_modulo(first_group, block, valid_groups);
    for (int local_rank = 0; local_rank < context->ranks_per_group; ++local_rank)
    {
      const unsigned char* record = cyclic_batch + (size_t)block * layout->group_bytes +
                                    (size_t)local_rank * layout->record_bytes;
      uint32_t packed_length = decode_length(record);
      int source_rank = global_rank(context, source_group, local_rank);
      unsigned char* destination =
        (unsigned char*)recvbuf + (size_t)source_rank * layout->receive_stride_bytes;
      int position = 0;
      int error;

      if (packed_length > (uint32_t)layout->pack_capacity)
      {
        return MPI_ERR_TRUNCATE;
      }
      error =
        MPI_Unpack(record + ALLGATHER_RADIX_BATCH_RECORD_HEADER_BYTES,
                   (int)packed_length,
                   &position,
                   destination,
                   recvcount,
                   recvtype,
                   context->comm);
      if (error != MPI_SUCCESS)
      {
        return error;
      }
      if (position != (int)packed_length)
      {
        return MPI_ERR_TRUNCATE;
      }
    }
  }
  return MPI_SUCCESS;
}

/** @brief 一批中连续的一段组：从 first_group 开始的 group_count 个组（组号相对于本批）。 */
typedef struct
{
  int first_group;  ///< 段内第一个组（相对于本批的序号）
  int group_count;  ///< 段内的组数
} GroupSegment;

/**
 * @brief 把循环区间的有效部分拆成线性的段（最多两段）。
 *
 * 区间的定义与 valid_block_count() 相同：从 start 开始、长度为 length，按 group_size 折返，
 * 只保留 [0, valid_groups) 内的位置。不折返时最多一段；折返时第二段从 0 开始。
 *
 * @param[in]  start        区间起点（0 ≤ start < group_size）
 * @param[in]  length       区间长度（≤ group_size）
 * @param[in]  group_size   模数 B
 * @param[in]  valid_groups 本批的组数
 * @param[out] segments     得到的线性段
 * @return 段数（0、1 或 2）。
 */
static int interval_segments(int start,
                             int length,
                             int group_size,
                             int valid_groups,
                             GroupSegment segments[2])
{
  int first_length = min_int(length, group_size - start);
  int count = 0;
  if (start < valid_groups)
  {
    int valid = min_int(first_length, valid_groups - start);
    if (valid > 0)
    {
      segments[count++] = (GroupSegment){start, valid};
    }
  }
  if (length > first_length)
  {
    int valid = min_int(length - first_length, valid_groups);
    if (valid > 0)
    {
      segments[count++] = (GroupSegment){0, valid};
    }
  }
  return count;
}

/**
 * @brief 第 3 阶段（原生路径）：直接在 recvbuf 的最终位置上做 Bruck 交换。
 *
 * 第 2 阶段已经把每个进程最初持有的组块放在 recvbuf 中的最终地址。这里交换的逻辑区间与
 * gather_batch() 完全相同，只是区间在 B 处折返时最多拆成两条消息（见 interval_segments()）。
 * 不需要应用层打包、循环缓冲区、旋转或最后的结果拷贝。
 *
 * @param[in,out] context      上下文
 * @param[in]     layout       当前布局（group_bytes）
 * @param[in]     batch_begin  本批第一个组的组号
 * @param[in]     valid_groups 本批的组数
 * @param[in,out] receive_base 用户的 recvbuf
 */
static void gather_batch_direct(AllgatherRadixBatchContext* context,
                                const CallLayout* layout,
                                int batch_begin,
                                int valid_groups,
                                unsigned char* receive_base)
{
  int group_size = context->ranks_per_group;
  int distance = 1;

  while (distance < group_size)
  {
    int request_count = 0;
    int offset = distance;
    int partner_index = 1;

    while (partner_index < context->radix && offset < group_size)
    {
      int length = min_int(distance, group_size - offset);
      int source_local = add_modulo(context->local_rank, offset, group_size);
      int destination_local = subtract_modulo(context->local_rank, offset, group_size);
      int source = global_rank(context, context->group_index, source_local);
      int destination = global_rank(context, context->group_index, destination_local);
      GroupSegment receive_segments[2];
      GroupSegment send_segments[2];
      int receive_count =
        interval_segments(source_local, length, group_size, valid_groups, receive_segments);
      int send_count = interval_segments(context->local_rank, length, group_size, valid_groups, send_segments);

      for (int segment = 0; segment < receive_count; ++segment)
      {
        size_t byte_offset = (size_t)(batch_begin + receive_segments[segment].first_group) *
                             layout->group_bytes;
        int bytes =
          (int)((size_t)receive_segments[segment].group_count * layout->group_bytes);
        require_communication(
          context->comm,
          MPI_Irecv(receive_base + byte_offset,
                    bytes,
                    MPI_BYTE,
                    source,
                    ALLGATHER_RADIX_BATCH_TAG_BRUCK,
                    context->comm,
                    &context->workspace.requests[request_count++]));
      }
      for (int segment = 0; segment < send_count; ++segment)
      {
        size_t byte_offset = (size_t)(batch_begin + send_segments[segment].first_group) *
                             layout->group_bytes;
        int bytes = (int)((size_t)send_segments[segment].group_count * layout->group_bytes);
        require_communication(
          context->comm,
          MPI_Isend(receive_base + byte_offset,
                    bytes,
                    MPI_BYTE,
                    destination,
                    ALLGATHER_RADIX_BATCH_TAG_BRUCK,
                    context->comm,
                    &context->workspace.requests[request_count++]));
      }
      if (offset > group_size - 1 - distance)
      {
        break;
      }
      offset += distance;
      ++partner_index;
    }
    require_communication(
      context->comm,
      MPI_Waitall(request_count,
                  context->workspace.requests,
                  MPI_STATUSES_IGNORE));
    distance = next_distance(distance, context->radix, group_size);
  }
}

/**
 * @brief 判断类型能否走原生路径（以 MPI_BYTE 直接传输）：是否为布局稠密的预定义类型。
 *
 * 只接受 MPI 预定义类型（combiner 为 MPI_COMBINER_NAMED），并要求 lower bound 与 true lower
 * bound 都为 0、extent 与 true extent 都等于类型大小。派生类型即使边界相同也不接受
 * （例如 hindexed 可能颠倒元素顺序）。
 *
 * @param[in]  type  MPI 数据类型
 * @param[out] dense 是稠密的预定义类型时为 1，否则为 0
 * @return MPI_SUCCESS 或类型查询的错误码（此时 dense 为 0）。
 */
static int is_dense_type(MPI_Datatype type, int* dense)
{
  int integer_count = 0;
  int address_count = 0;
  int datatype_count = 0;
  int combiner = 0;
  int bytes = 0;
  MPI_Aint lower_bound = 0;
  MPI_Aint extent = 0;
  MPI_Aint true_lower_bound = 0;
  MPI_Aint true_extent = 0;
  int error =
    MPI_Type_get_envelope(type, &integer_count, &address_count, &datatype_count, &combiner);
  *dense = 0;
  if (error != MPI_SUCCESS || combiner != MPI_COMBINER_NAMED)
  {
    return error;
  }
  /* Equal bounds alone do not prove typemap order: hindexed may reverse it. */
  error = MPI_Type_size(type, &bytes);
  if (error == MPI_SUCCESS)
  {
    error = MPI_Type_get_extent(type, &lower_bound, &extent);
  }
  if (error == MPI_SUCCESS)
  {
    error = MPI_Type_get_true_extent(type, &true_lower_bound, &true_extent);
  }
  if (error == MPI_SUCCESS)
  {
    *dense = bytes >= 0 && lower_bound == 0 && true_lower_bound == 0 && extent == bytes &&
             true_extent == bytes;
  }
  return error;
}

/**
 * @brief 计算原生路径的布局。
 *
 * 记录就是 payload_bytes 字节的用户数据，组块 = B 个记录。类型稠密时 receive_stride 等于
 * payload_bytes，所以一个组块在 recvbuf 中是连续的。工作区只需 normalized_group 和
 * ordered_group 两个组块；第 3 阶段最大的单条消息为 min(G, B) 个组块，不能超过 INT_MAX。
 *
 * @param[in]  context        上下文
 * @param[in]  payload_bytes  每个进程贡献的字节数（所有进程相同）
 * @param[in]  receive_stride recvbuf 中相邻贡献的间距（字节）
 * @param[out] layout         计算出的布局
 * @return MPI_SUCCESS 或 MPI_ERR_COUNT。
 */
static int make_native_layout(const AllgatherRadixBatchContext* context,
                              size_t payload_bytes,
                              size_t receive_stride,
                              CallLayout* layout)
{
  size_t maximum_message;
  int error;

  memset(layout, 0, sizeof(*layout));
  layout->native_bytes = 1;
  layout->record_bytes = payload_bytes;
  layout->receive_stride_bytes = receive_stride;
  error = checked_multiply((size_t)context->ranks_per_group, payload_bytes, &layout->group_bytes);
  if (error != MPI_SUCCESS || layout->group_bytes > INT_MAX)
  {
    return MPI_ERR_COUNT;
  }
  error = checked_multiply((size_t)min_int(context->group_count, context->ranks_per_group),
                           layout->group_bytes,
                           &maximum_message);
  if (error != MPI_SUCCESS || maximum_message > INT_MAX)
  {
    return MPI_ERR_COUNT;
  }
  return checked_multiply(2, layout->group_bytes, &layout->storage_bytes);
}

/**
 * @brief 集合地绑定个数、类型和就地模式，并一次性预留工作区（应在计时循环之外调用）。
 *
 * 步骤：本地检查参数（见 validate_call()）→ 协商错误 → 检查所有进程的“数据是否为空”、就地模式
 * 和 options 一致 → 选择数据路径：请求了 ALLGATHER_RADIX_BATCH_NATIVE_FAST、所有进程的发送和
 * 接收类型都是稠密预定义类型（见 is_dense_type()）且字节数相同时用原生布局，否则用打包记录布局
 * （记录容量取所有进程 MPI_Pack_size 的最大值）→ 预留工作区 → 保存绑定。
 *
 * 失败时保留上一次成功的绑定。绑定的类型句柄在下一次 prepare 或销毁之前必须保持有效。
 *
 * @param[in,out] context   上下文
 * @param[in]     sendcount 发送个数（就地模式下忽略）
 * @param[in]     sendtype  发送类型（就地模式下忽略）
 * @param[in]     recvcount 每个进程的接收个数
 * @param[in]     recvtype  接收类型
 * @param[in]     in_place  1 表示之后的调用使用 MPI_IN_PLACE，0 表示不使用
 * @param[in]     options   ALLGATHER_RADIX_BATCH_PORTABLE 或 ALLGATHER_RADIX_BATCH_NATIVE_FAST
 * @return MPI_SUCCESS；context 为 NULL、in_place 或 options 非法、或各进程不一致时返回 MPI_ERR_ARG；
 *         有的进程数据为空而有的不为空、或原生路径下各进程字节数不同时返回 MPI_ERR_TRUNCATE；
 *         以及 validate_call()、布局计算和 reserve_workspace() 的错误码。
 * @note 集合操作。
 */
int Allgather_radix_batch_prepare(AllgatherRadixBatchContext* context,
                                  int sendcount,
                                  MPI_Datatype sendtype,
                                  int recvcount,
                                  MPI_Datatype recvtype,
                                  int in_place,
                                  unsigned options)
{
  unsigned char dummy = 0;
  const void* placeholder = in_place ? MPI_IN_PLACE : &dummy;
  unsigned long long payload_bytes = 0;
  unsigned long long minimum_bytes = 0;
  unsigned long long maximum_bytes = 0;
  size_t receive_stride = 0;
  int local_pack_bound = 0;
  int maximum_pack_bound = 0;
  int local_dense = 0;
  int receive_dense = 0;
  int all_dense = 0;
  int modes[3];
  int minimum_modes[3];
  int maximum_modes[3];
  PreparedCall prepared;
  int error;
  int local_error;

  if (context == NULL)
  {
    return MPI_ERR_ARG;
  }
  memset(&prepared, 0, sizeof(prepared));
  local_error = validate_call(context,
                              placeholder,
                              sendcount,
                              sendtype,
                              &dummy,
                              recvcount,
                              recvtype,
                              &local_pack_bound,
                              &payload_bytes,
                              &receive_stride);
  if ((in_place != 0 && in_place != 1) ||
      (options & ~(unsigned)ALLGATHER_RADIX_BATCH_NATIVE_FAST) != 0)
  {
    local_error = MPI_ERR_ARG;
  }
  if (local_error == MPI_SUCCESS && (options & ALLGATHER_RADIX_BATCH_NATIVE_FAST) != 0)
  {
    local_error = is_dense_type(in_place ? recvtype : sendtype, &local_dense);
    if (local_error == MPI_SUCCESS)
    {
      local_error = is_dense_type(recvtype, &receive_dense);
      local_dense = local_dense && receive_dense;
    }
  }
  error = agree_status(context, local_error);
  if (error != MPI_SUCCESS)
  {
    return error;
  }
  modes[0] = payload_bytes != 0;
  modes[1] = in_place;
  modes[2] = (int)options;
  require_communication(
    context->comm,
    MPI_Allreduce(modes,
                  minimum_modes,
                  3,
                  MPI_INT,
                  MPI_MIN,
                  context->comm));
  require_communication(
    context->comm,
    MPI_Allreduce(modes,
                  maximum_modes,
                  3,
                  MPI_INT,
                  MPI_MAX,
                  context->comm));
  if (minimum_modes[0] != maximum_modes[0])
  {
    return MPI_ERR_TRUNCATE;
  }
  if (minimum_modes[1] != maximum_modes[1] || minimum_modes[2] != maximum_modes[2])
  {
    return MPI_ERR_ARG;
  }
  require_communication(
    context->comm, MPI_Allreduce(&local_dense, &all_dense, 1, MPI_INT, MPI_MIN, context->comm));
  if (payload_bytes != 0 && all_dense)
  {
    require_communication(
      context->comm,
      MPI_Allreduce(&payload_bytes, &minimum_bytes, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, context->comm));

    require_communication(
      context->comm,
      MPI_Allreduce(&payload_bytes, &maximum_bytes, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, context->comm));
    if (minimum_bytes != maximum_bytes)
    {
      return MPI_ERR_TRUNCATE;
    }
    local_error =
      make_native_layout(context, (size_t)payload_bytes, receive_stride, &prepared.layout);
  }
  else if (payload_bytes != 0)
  {
    require_communication(context->comm, MPI_Allreduce(&local_pack_bound, &maximum_pack_bound, 1, MPI_INT, MPI_MAX, context->comm));
    local_error =
      make_call_layout(context, maximum_pack_bound, receive_stride, &prepared.layout);
  }
  error = agree_status(context, local_error);
  if (error != MPI_SUCCESS)
  {
    return error;
  }
  if (payload_bytes != 0)
  {
    error = reserve_workspace(context, prepared.layout.storage_bytes);
    if (error != MPI_SUCCESS)
    {
      return error;
    }
  }
  prepared.ready = 1;
  prepared.empty = payload_bytes == 0;
  prepared.in_place = in_place;
  prepared.sendcount = sendcount;
  prepared.sendtype = sendtype;
  prepared.recvcount = recvcount;
  prepared.recvtype = recvtype;
  context->prepared = prepared;
  return MPI_SUCCESS;
}

/**
 * @brief 已 prepare 时的执行路径（热路径）：不分配内存、不查询类型、不做错误协商。
 *
 * 参数必须与 prepare 时的绑定一致、缓冲区指针必须有效，否则直接中止（不能让其他进程卡在
 * 通信中）。依次执行：（可移植路径）打包本进程记录 → 第 1 阶段 gather_within_group() →
 * 第 2 阶段 distribute_group_blocks() → 对每一批执行第 3 阶段（原生路径 gather_batch_direct()；
 * 可移植路径 gather_batch() 后接 unpack_batch()）。
 *
 * @param[in,out] context   已 prepare 的上下文
 * @param[in]     sendbuf   发送缓冲区或 MPI_IN_PLACE
 * @param[in]     sendcount 发送个数
 * @param[in]     sendtype  发送类型
 * @param[out]    recvbuf   接收缓冲区
 * @param[in]     recvcount 每个进程的接收个数
 * @param[in]     recvtype  接收类型
 * @return MPI_SUCCESS（任何错误都会中止）。
 */
static int execute_prepared(AllgatherRadixBatchContext* context,
                            const void* sendbuf,
                            int sendcount,
                            MPI_Datatype sendtype,
                            void* recvbuf,
                            int recvcount,
                            MPI_Datatype recvtype)
{
  const PreparedCall* prepared = &context->prepared;
  const CallLayout* layout = &prepared->layout;
  const void* local_record;
  BufferViews views;
  int batch_begin = 0;
  int batch_index = 0;
  int in_place = sendbuf == MPI_IN_PLACE;
  int binding_matches =
    in_place == prepared->in_place && recvcount == prepared->recvcount &&
    recvtype == prepared->recvtype &&
    (in_place || (sendcount == prepared->sendcount && sendtype == prepared->sendtype));

  require_communication(context->comm, binding_matches ? MPI_SUCCESS : MPI_ERR_ARG);
  if (prepared->empty)
  {
    return MPI_SUCCESS;
  }
  require_communication(context->comm,
                        recvbuf != NULL && recvbuf != MPI_BOTTOM && recvbuf != MPI_IN_PLACE &&
                            (in_place || (sendbuf != NULL && sendbuf != MPI_BOTTOM))
                          ? MPI_SUCCESS
                          : MPI_ERR_BUFFER);
  views = buffer_views(context, layout);
  if (layout->native_bytes)
  {
    local_record = in_place ? (unsigned char*)recvbuf +
                                (size_t)context->rank * layout->receive_stride_bytes
                            : sendbuf;
  }
  else
  {
    require_communication(
      context->comm,
      pack_local_record(context,
                        layout,
                        views.local_record,
                        sendbuf,
                        sendcount,
                        sendtype,
                        recvbuf,
                        recvcount,
                        recvtype));
    local_record = views.local_record;
  }
  gather_within_group(context, layout, &views, local_record);
  distribute_group_blocks(context, layout, &views, recvbuf);
  while (batch_begin < context->group_count)
  {
    int batch_groups = min_int(context->ranks_per_group, context->group_count - batch_begin);
    if (layout->native_bytes)
    {
      gather_batch_direct(context, layout, batch_begin, batch_groups, recvbuf);
    }
    else
    {
      const unsigned char* assigned =
        views.assigned_blocks + (size_t)batch_index * layout->group_bytes;
      gather_batch(context, layout, assigned, batch_groups, views.cyclic_batch);
      require_communication(
        context->comm,
        unpack_batch(context,
                     layout,
                     views.cyclic_batch,
                     batch_begin,
                     batch_groups,
                     recvbuf,
                     recvcount,
                     recvtype));
    }
    batch_begin += batch_groups;
    ++batch_index;
  }
  return MPI_SUCCESS;
}

/**
 * @brief 集合地执行一次 Allgather：recvbuf 中按 rank 顺序得到每个进程的贡献。
 *
 * 语义与 MPI_Allgather 相同：每个发送方的类型签名必须与接收方的 (recvcount, recvtype) 一致。
 * 支持非连续和 resized 类型、负的位移、空数据，以及所有进程同时使用 MPI_IN_PLACE；
 * 不支持 MPI_BOTTOM/绝对地址缓冲区和组间通信子。
 *
 * - 已 prepare：转到 execute_prepared()，参数必须与绑定一致，任何错误都会中止。
 * - 未 prepare：每次调用都要检查参数、协商、计算打包布局并按需扩大工作区，只走可移植路径；
 *   参数、内存、打包和解包错误在所有进程上返回相同的错误码（解包出错的进程会继续参与剩余
 *   的通信，最后再协商）。
 *
 * @param[in,out] context   上下文
 * @param[in]     sendbuf   发送缓冲区或 MPI_IN_PLACE
 * @param[in]     sendcount 发送个数（就地模式下忽略）
 * @param[in]     sendtype  发送类型（就地模式下忽略）
 * @param[out]    recvbuf   接收缓冲区：第 i 个进程的贡献位于偏移 i × recvcount × extent(recvtype) 处
 * @param[in]     recvcount 每个进程的接收个数
 * @param[in]     recvtype  接收类型
 * @return MPI_SUCCESS 或错误码（见上）；出错后 recvbuf 的内容不确定。
 * @note 集合操作。上下文不可重入，调用与销毁必须串行。
 */
int Allgather_radix_batch_ctx(AllgatherRadixBatchContext* context,
                              const void* sendbuf,
                              int sendcount,
                              MPI_Datatype sendtype,
                              void* recvbuf,
                              int recvcount,
                              MPI_Datatype recvtype)
{
  int local_pack_bound = 0;
  int maximum_pack_bound = 0;
  unsigned long long payload_bytes = 0;
  size_t receive_stride = 0;
  int modes[2];
  int minimum_modes[2];
  int maximum_modes[2];
  int local_error;
  int error;
  int batch_begin = 0;
  int batch_index = 0;
  CallLayout layout;
  BufferViews views;

  if (context == NULL)
  {
    return MPI_ERR_ARG;
  }
  if (context->prepared.ready)
  {
    return execute_prepared(context,
                            sendbuf,
                            sendcount,
                            sendtype,
                            recvbuf,
                            recvcount,
                            recvtype);
  }
  local_error = validate_call(context,
                              sendbuf,
                              sendcount,
                              sendtype,
                              recvbuf,
                              recvcount,
                              recvtype,
                              &local_pack_bound,
                              &payload_bytes,
                              &receive_stride);
  error = agree_status(context, local_error);
  if (error != MPI_SUCCESS)
  {
    return error;
  }
  modes[0] = payload_bytes != 0;
  modes[1] = sendbuf == MPI_IN_PLACE;
  require_communication(context->comm,
                        MPI_Allreduce(modes, minimum_modes, 2, MPI_INT, MPI_MIN, context->comm));
  require_communication(context->comm,
                        MPI_Allreduce(modes, maximum_modes, 2, MPI_INT, MPI_MAX, context->comm));
  if (minimum_modes[0] != maximum_modes[0])
  {
    return MPI_ERR_TRUNCATE;
  }
  if (minimum_modes[1] != maximum_modes[1])
  {
    return MPI_ERR_ARG;
  }
  if (payload_bytes == 0)
  {
    return MPI_SUCCESS;
  }
  require_communication(context->comm, MPI_Allreduce(&local_pack_bound, &maximum_pack_bound, 1, MPI_INT, MPI_MAX, context->comm));
  local_error = make_call_layout(context, maximum_pack_bound, receive_stride, &layout);
  error = agree_status(context, local_error);
  if (error != MPI_SUCCESS)
  {
    return error;
  }
  error = reserve_workspace(context, layout.storage_bytes);
  if (error != MPI_SUCCESS)
  {
    return error;
  }
  views = buffer_views(context, &layout);
  local_error = pack_local_record(context, &layout, views.local_record, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype);
  error = agree_status(context, local_error);
  if (error != MPI_SUCCESS)
  {
    return error;
  }

  gather_within_group(context, &layout, &views, views.local_record);
  distribute_group_blocks(context, &layout, &views, recvbuf);

  local_error = MPI_SUCCESS;
  while (batch_begin < context->group_count)
  {
    int batch_groups = min_int(context->ranks_per_group, context->group_count - batch_begin);
    const unsigned char* assigned_block =
      views.assigned_blocks + (size_t)batch_index * layout.group_bytes;

    gather_batch(context, &layout, assigned_block, batch_groups, views.cyclic_batch);
    /* Keep participating after a local unpack error; all peers finish first. */
    if (local_error == MPI_SUCCESS)
    {
      local_error = unpack_batch(context, &layout, views.cyclic_batch, batch_begin, batch_groups, recvbuf, recvcount, recvtype);
    }
    batch_begin += batch_groups;
    ++batch_index;
  }
  return agree_status(context, local_error);
}

/**
 * @brief 集合地销毁上下文：释放工作区和请求数组，并释放私有通信子。
 * @param[in,out] context 指向上下文指针的指针；成功后 *context 置为 NULL
 * @return MPI_SUCCESS；context 或 *context 为 NULL 时返回 MPI_ERR_ARG。
 * @note 集合操作（MPI_Comm_free）。
 */
int Allgather_radix_batch_context_destroy(AllgatherRadixBatchContext** context)
{
  AllgatherRadixBatchContext* owned;
  MPI_Comm comm;

  if (context == NULL || *context == NULL)
  {
    return MPI_ERR_ARG;
  }
  owned = *context;
  comm = owned->comm;
  free(owned->workspace.base);
  free(owned->workspace.requests);
  free(owned);
  *context = NULL;
  require_communication(comm, MPI_Comm_free(&comm));
  return MPI_SUCCESS;
}

/**
 * @brief 便捷入口 Allgather_radix_batch() 缓存的上下文，键为 (comm 句柄, radix, ranks_per_group)。
 *
 * 创建并 prepare 一个上下文需要一次 MPI_Comm_dup 和十几次集合操作，所以只在某个键第一次
 * 调用时创建；只有绑定（个数、类型、就地模式）改变时才重新 prepare。所有进程按相同的顺序
 * 调用，因此缓存的命中与未命中在各进程上一致。
 */
typedef struct
{
  MPI_Comm comm;                        ///< 用户传入的通信子句柄（键）
  int radix;                            ///< 基数 k（键）
  int ranks_per_group;                  ///< 每组进程数 B（键）
  AllgatherRadixBatchContext* context;  ///< 缓存的上下文
} CachedContext;

static CachedContext* cached_contexts;  ///< 缓存数组
static int cached_context_count;        ///< 已缓存的上下文个数
static int cached_context_capacity;     ///< 缓存数组的容量

/**
 * @brief 查找或创建 (comm, radix, ranks_per_group) 对应的缓存上下文。
 *
 * 命中时直接返回；未命中时按需扩大缓存数组（初始 16 个，之后翻倍），再集合地调用
 * Allgather_radix_batch_context_create() 创建新上下文并加入缓存。
 *
 * @param[in]  comm            用户的通信子
 * @param[in]  radix           基数 k
 * @param[in]  ranks_per_group 每组进程数 B
 * @param[out] context         找到或新建的上下文
 * @return MPI_SUCCESS；缓存数组扩大失败时返回 MPI_ERR_NO_MEM；以及创建上下文的错误码。
 * @note 缓存数组扩大失败（MPI_ERR_NO_MEM）只在本进程返回，不与其他进程协商。
 */
static int cached_context(MPI_Comm comm,
                          int radix,
                          int ranks_per_group,
                          AllgatherRadixBatchContext** context)
{
  CachedContext* grown;
  int error;

  for (int index = 0; index < cached_context_count; ++index)
  {
    const CachedContext* entry = &cached_contexts[index];
    if (entry->comm == comm && entry->radix == radix && entry->ranks_per_group == ranks_per_group)
    {
      *context = entry->context;
      return MPI_SUCCESS;
    }
  }
  if (cached_context_count == cached_context_capacity)
  {
    int capacity = cached_context_capacity ? 2 * cached_context_capacity : 16;
    grown = realloc(cached_contexts, (size_t)capacity * sizeof(*grown));
    if (grown == NULL)
    {
      return MPI_ERR_NO_MEM;
    }
    cached_contexts = grown;
    cached_context_capacity = capacity;
  }
  error = Allgather_radix_batch_context_create(comm, radix, ranks_per_group, context);
  if (error != MPI_SUCCESS)
  {
    return error;
  }
  cached_contexts[cached_context_count++] =
    (CachedContext){comm, radix, ranks_per_group, *context};
  return MPI_SUCCESS;
}

/**
 * @brief 便捷入口，参数顺序与原始算法相同：以基数 k = radix、每组 B = ranks_per_group 个进程做 Allgather。
 *
 * sendcount/datatype 同时描述每个接收块（就地模式也是如此）。某个 (comm, radix, ranks_per_group)
 * 第一次调用时集合地创建并缓存上下文；绑定（个数、类型、就地模式）与上一次不同时，以
 * ALLGATHER_RADIX_BATCH_NATIVE_FAST 重新 prepare（集合操作）；其余调用只做通信。
 * 计时时应丢弃这些第一次调用。
 *
 * @param[in]  sendbuf         发送缓冲区或 MPI_IN_PLACE
 * @param[in]  sendcount       每个进程贡献的元素个数
 * @param[in]  datatype        元素类型（发送和接收相同）
 * @param[out] recvbuf         接收缓冲区，按 rank 顺序存放 P × sendcount 个元素
 * @param[in]  comm            通信子
 * @param[in]  radix           基数 k（≥ 2）
 * @param[in]  ranks_per_group 每组进程数 B（整除进程数）
 * @return MPI_SUCCESS 或错误码。
 * @note 缓存以 comm 句柄为键：释放一个在这里用过的通信子之前，先调用
 *       Allgather_radix_batch_release_cached()。原生传输要求同构的主机内存。
 */
int Allgather_radix_batch(
  const void* sendbuf,
  int sendcount,
  MPI_Datatype datatype,
  void* recvbuf,
  MPI_Comm comm,
  int radix,
  int ranks_per_group)
{
  AllgatherRadixBatchContext* context = NULL;
  const PreparedCall* prepared;
  int in_place = sendbuf == MPI_IN_PLACE;
  int error = cached_context(comm, radix, ranks_per_group, &context);

  if (error != MPI_SUCCESS) return error;

  prepared = &context->prepared;
  if (!prepared->ready || prepared->in_place != in_place ||
      prepared->recvcount != sendcount || prepared->recvtype != datatype ||
      (!in_place && (prepared->sendcount != sendcount || prepared->sendtype != datatype)))
  {
    error = Allgather_radix_batch_prepare(context,
                                          sendcount,
                                          datatype,
                                          sendcount,
                                          datatype,
                                          in_place,
                                          ALLGATHER_RADIX_BATCH_NATIVE_FAST);
    if (error != MPI_SUCCESS) return error;
  }
  return Allgather_radix_batch_ctx(context,
                                   sendbuf,
                                   sendcount,
                                   datatype,
                                   recvbuf,
                                   sendcount,
                                   datatype);
}

/**
 * @brief 集合地销毁 Allgather_radix_batch() 缓存的所有上下文，并清空缓存。
 * @note 集合操作：会释放每个缓存上下文的私有通信子，所有进程都必须调用。
 */
void Allgather_radix_batch_release_cached(void)
{
  for (int index = 0; index < cached_context_count; ++index)
  {
    (void)Allgather_radix_batch_context_destroy(&cached_contexts[index].context);
  }
  free(cached_contexts);
  cached_contexts = NULL;
  cached_context_count = 0;
  cached_context_capacity = 0;
}