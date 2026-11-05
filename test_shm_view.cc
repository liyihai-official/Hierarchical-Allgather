// Tests of the Allgather_shm_view contract. Exit code 0 when every rank passes.
//
//   test_shm_view <ranks per node>
//
// The slow reader case is the important one: a process keeps reading its result while the others
// of its node are already in their next call. It must fail if that next call stops waiting.
#include <mpi.h>
#include <setjmp.h>
#include <signal.h>
#include <time.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "Allgather_shm.h"

namespace
{

int g_rank = 0;
int g_nprocs = 0;
int g_ppn = 0;
long g_failures = 0;
sigjmp_buf g_fault_jump;

void expect(bool ok, const char* what)
{
  if (!ok)
  {
    ++g_failures;
    if (g_failures <= 3) std::fprintf(stderr, "rank %d: FAILED %s\n", g_rank, what);
  }
}

// Every iteration changes every element, so data of a later call is always detected.
std::vector<int> contribution(int iteration, int count)
{
  std::vector<int> send(count);
  for (int i = 0; i < count; ++i) send[i] = iteration + g_rank * count + i;
  return send;
}

bool is_correct(const int* result, int iteration, int count)
{
  const long long elements = static_cast<long long>(g_nprocs) * count;
  for (long long i = 0; i < elements; ++i)
  {
    if (result[i] != iteration + static_cast<int>(i)) return false;
  }
  return true;
}

const int* view(int iteration, int count)
{
  const std::vector<int> send = contribution(iteration, count);
  const void* shared = nullptr;
  const int error = Allgather_shm_view(send.data(), count, MPI_INT, &shared, MPI_COMM_WORLD, g_ppn);
  expect(error == MPI_SUCCESS, "view returned an error");
  return static_cast<const int*>(shared);
}

void test_content()
{
  for (int count : {1, 7, 1024, 4096, 16, 1})
  {
    for (int iteration = 0; iteration < 5; ++iteration)
    {
      expect(is_correct(view(iteration, count), iteration, count), "content");
    }
  }
}

void test_slow_reader()
{
  const timespec delay{0, 2000000};
  for (int iteration = 0; iteration < 300; ++iteration)
  {
    const int* result = view(iteration, 256);
    if (g_rank % g_ppn == iteration % g_ppn) nanosleep(&delay, nullptr);
    expect(is_correct(result, iteration, 256), "slow reader saw data of a later call");
  }
}

void test_mixed_with_copy()
{
  const int count = 64;
  std::vector<int> recv(static_cast<size_t>(g_nprocs) * count);
  for (int iteration = 0; iteration < 20; ++iteration)
  {
    expect(is_correct(view(iteration, count), iteration, count), "view between copies");
    const std::vector<int> send = contribution(iteration + 1, count);
    const int error = Allgather_shm(send.data(), count, MPI_INT, recv.data(), MPI_COMM_WORLD, g_ppn);
    expect(error == MPI_SUCCESS, "copy version returned an error");
    expect(is_correct(recv.data(), iteration + 1, count), "copy between views");
  }
}

void on_fault(int)
{
  siglongjmp(g_fault_jump, 1);
}

void test_read_only()
{
  const int count = 8;
  const int* result = view(3, count);
  struct sigaction action = {};
  struct sigaction previous = {};
  action.sa_handler = on_fault;
  sigemptyset(&action.sa_mask);
  sigaction(SIGSEGV, &action, &previous);
  sigaction(SIGBUS, &action, nullptr);
  bool faulted = false;
  if (sigsetjmp(g_fault_jump, 1) == 0)
  {
    *const_cast<volatile int*>(result) = -1;
  }
  else
  {
    faulted = true;
  }
  sigaction(SIGSEGV, &previous, nullptr);
  signal(SIGBUS, SIG_DFL);
  expect(faulted, "a write to the result did not fault");
  expect(is_correct(result, 3, count), "content after the refused write");
}

void test_refused_arguments()
{
  const int value = 0;
  const void* shared = &value;
  expect(Allgather_shm_view(MPI_IN_PLACE, 1, MPI_INT, &shared, MPI_COMM_WORLD, g_ppn) == MPI_ERR_BUFFER,
         "MPI_IN_PLACE not refused");
  expect(Allgather_shm_view(&value, 1, MPI_INT, nullptr, MPI_COMM_WORLD, g_ppn) == MPI_ERR_ARG,
         "NULL result not refused");
}

void test_empty()
{
  const int value = 0;
  const void* shared = &value;
  expect(Allgather_shm_view(&value, 0, MPI_INT, &shared, MPI_COMM_WORLD, g_ppn) == MPI_SUCCESS,
         "empty call returned an error");
  expect(shared == nullptr, "empty call did not return NULL");
  expect(is_correct(view(2, 16), 2, 16), "content after an empty call");
}

// Releasing while a result is held must not hang, and the view must work again afterwards.
void test_release()
{
  expect(is_correct(view(1, 32), 1, 32), "content before release");
  Allgather_shm_release_cached();
  expect(is_correct(view(2, 32), 2, 32), "content after release");
}

}  // namespace

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nprocs);
  g_ppn = argc > 1 ? std::atoi(argv[1]) : 0;
  if (g_ppn < 1)
  {
    if (g_rank == 0) std::fprintf(stderr, "usage: test_shm_view <ranks per node>\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  const struct
  {
    const char* name;
    void (*run)();
  } cases[] = {{"content", test_content},
               {"slow_reader", test_slow_reader},
               {"mixed_with_copy", test_mixed_with_copy},
               {"read_only", test_read_only},
               {"refused_arguments", test_refused_arguments},
               {"empty", test_empty},
               {"release", test_release}};

  long total = 0;
  for (const auto& test : cases)
  {
    const long before = g_failures;
    test.run();
    long failures = g_failures - before;
    MPI_Allreduce(MPI_IN_PLACE, &failures, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    total += failures;
    if (g_rank == 0) std::printf("%s %s: %ld failures\n", failures == 0 ? "PASS" : "FAIL", test.name, failures);
  }
  if (g_rank == 0) std::printf("%s nprocs=%d ppn=%d\n", total == 0 ? "ALL PASS" : "SOME FAILED", g_nprocs, g_ppn);

  Allgather_shm_release_cached();
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
