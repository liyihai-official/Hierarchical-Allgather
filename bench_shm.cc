// Compares MPI_Allgather, Allgather_radix_batch and Allgather_shm in one allocation.
//
//   bench_shm nodes=<n> ppn=<p> [b=<group size, default ppn>] [out=<csv>]
//             [counts=1,2,4] [radices=2,4,8] [reps=<fixed reps>] [warmup=3]
//             [max_seconds=<stop before the next configuration>] [inplace=1] [diag=1]
//             [view=1: also run Allgather_shm_view] [read_pass=1: time one read of the result]
//
// Without reps=, the number of repetitions drops with size: 50 up to 1024 elements,
// 20 up to 8192, then 10. Warmup calls absorb setup and first-use costs and are not logged.
// time_barrier is the method of main.cc (rank 0, including a closing barrier); time_max is
// the slowest rank's own duration without that barrier. read_max is the slowest rank's time to
// sum the whole result after a barrier: Allgather_shm_view skips a copy that other algorithms
// pay for inside the call, so comparing them fairly needs the cost of using the result too.
#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "Allgather_radix_batch.h"
#include "Allgather_shm.h"

namespace
{

enum class Kind
{
  MpiAllgather,
  RadixBatch,
  NodeShm,
  NodeView
};

struct Algorithm
{
  Kind kind;
  int k;
};

struct Options
{
  int nodes = 1;
  int ppn = 1;
  int b = 0;
  int warmup = 3;
  int reps = 0;
  int inplace = 0;
  int diag = 0;
  int view = 0;
  int read_pass = 0;
  double max_seconds = 0;
  std::string out = "bench_shm.csv";
  std::vector<int> counts;
  std::vector<int> radices{2, 4, 8, 16, 32, 64, 102};
};

const char* name_of(Kind kind)
{
  switch (kind)
  {
    case Kind::MpiAllgather: return "mpi_allgather";
    case Kind::RadixBatch: return "radix_batch";
    case Kind::NodeShm: return "node_shm";
    case Kind::NodeView: return "node_view";
  }
  return "unknown";
}

std::vector<int> parse_list(const char* text)
{
  std::vector<int> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ','))
  {
    values.push_back(std::atoi(item.c_str()));
  }
  return values;
}

Options parse_options(int argc, char** argv)
{
  Options options;
  for (int i = 1; i < argc; ++i)
  {
    const char* arg = argv[i];
    const char* eq = std::strchr(arg, '=');
    if (eq == nullptr)
    {
      std::cerr << "Unknown argument: " << arg << "\n";
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    std::string key(arg, eq - arg);
    const char* value = eq + 1;
    if (key == "nodes") options.nodes = std::atoi(value);
    else if (key == "ppn") options.ppn = std::atoi(value);
    else if (key == "b") options.b = std::atoi(value);
    else if (key == "warmup") options.warmup = std::atoi(value);
    else if (key == "reps") options.reps = std::atoi(value);
    else if (key == "inplace") options.inplace = std::atoi(value);
    else if (key == "diag") options.diag = std::atoi(value);
    else if (key == "view") options.view = std::atoi(value);
    else if (key == "read_pass") options.read_pass = std::atoi(value);
    else if (key == "max_seconds") options.max_seconds = std::atof(value);
    else if (key == "out") options.out = value;
    else if (key == "counts") options.counts = parse_list(value);
    else if (key == "radices") options.radices = parse_list(value);
    else
    {
      std::cerr << "Unknown argument: " << arg << "\n";
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
  }
  if (options.b == 0) options.b = options.ppn;
  if (options.counts.empty())
  {
    for (int count = 1; count <= 32768; count *= 2) options.counts.push_back(count);
  }
  return options;
}

int reps_for(const Options& options, int count)
{
  if (options.reps > 0) return options.reps;
  if (count <= 1024) return 50;
  if (count <= 8192) return 20;
  return 10;
}

// Sets *result to where this call left the data.
int call(const Algorithm& algorithm, const void* send, int count, int* recv, int b, const int** result)
{
  *result = recv;
  switch (algorithm.kind)
  {
    case Kind::MpiAllgather:
      return MPI_Allgather(send, count, MPI_INT, recv, count, MPI_INT, MPI_COMM_WORLD);
    case Kind::RadixBatch:
      return Allgather_radix_batch(send, count, MPI_INT, recv, MPI_COMM_WORLD, algorithm.k, b);
    case Kind::NodeShm:
      return Allgather_shm(send, count, MPI_INT, recv, MPI_COMM_WORLD, b);
    case Kind::NodeView:
    {
      const void* shared = nullptr;
      const int error = Allgather_shm_view(send, count, MPI_INT, &shared, MPI_COMM_WORLD, b);
      *result = static_cast<const int*>(shared);
      return error;
    }
  }
  return MPI_ERR_OTHER;
}

const double* stage_times(Kind kind)
{
  static const double none[3] = {0, 0, 0};
  switch (kind)
  {
    case Kind::RadixBatch: return Allgather_radix_batch_stage_times;
    case Kind::NodeShm:
    case Kind::NodeView: return Allgather_shm_stage_times;
    default: return none;
  }
}

// Every element of the result equals its own index: rank * count + i.
bool result_is_correct(const int* result, size_t size)
{
  for (size_t i = 0; i < size; ++i)
  {
    if (result[i] != static_cast<int>(i)) return false;
  }
  return true;
}

void run_algorithm(const Algorithm& algorithm,
                   const Options& options,
                   int count,
                   const std::vector<int>& send,
                   std::vector<int>& recv,
                   std::ofstream& csv,
                   int rank,
                   int nprocs)
{
  const int reps = reps_for(options, count);
  const size_t own_begin = static_cast<size_t>(rank) * count;

  for (int rep = -options.warmup; rep < reps; ++rep)
  {
    std::fill(recv.begin(), recv.end(), 0);
    if (options.inplace)
    {
      std::copy(send.begin(), send.end(), recv.begin() + own_begin);
    }
    const void* send_arg = options.inplace ? MPI_IN_PLACE : static_cast<const void*>(send.data());

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    const int* result = nullptr;
    const int error = call(algorithm, send_arg, count, recv.data(), options.b, &result);
    const double local_time = MPI_Wtime() - t0;
    MPI_Barrier(MPI_COMM_WORLD);
    const double barrier_time = MPI_Wtime() - t0;

    if (error != MPI_SUCCESS)
    {
      if (rank == 0)
      {
        char text[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, text, &length);
        std::cout << "SKIP " << name_of(algorithm.kind) << " count=" << count << " b=" << options.b
                  << ": " << text << std::endl;
      }
      return;
    }

    // Warmup calls read too: the first read of a mapping takes page faults.
    double read_time = 0;
    long long read_sum = 0;
    if (options.read_pass)
    {
      MPI_Barrier(MPI_COMM_WORLD);
      const double t1 = MPI_Wtime();
      read_sum = std::accumulate(result, result + recv.size(), 0LL);
      read_time = MPI_Wtime() - t1;
    }
    if (rep < 0) continue;

    const double* stages = stage_times(algorithm.kind);
    const double local[5] = {local_time, stages[0], stages[1], stages[2], read_time};
    double maximum[5] = {0, 0, 0, 0, 0};
    double sum[3] = {0, 0, 0};
    const long long elements = static_cast<long long>(recv.size());
    const int correct = result_is_correct(result, recv.size()) &&
                        (!options.read_pass || read_sum == elements * (elements - 1) / 2);
    int all_correct = 0;
    MPI_Reduce(local, maximum, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(local + 1, sum, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&correct, &all_correct, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
      csv << name_of(algorithm.kind) << ',' << algorithm.k << ',' << options.b << ','
          << options.nodes << ',' << options.ppn << ',' << nprocs << ',' << count << ',' << rep << ','
          << barrier_time << ',' << maximum[0] << ',' << maximum[1] << ',' << maximum[2] << ','
          << maximum[3] << ',' << sum[0] / nprocs << ',' << sum[1] / nprocs << ',' << sum[2] / nprocs
          << ',' << all_correct << ',' << maximum[4] << '\n';
    }
  }
  if (rank == 0) csv.flush();
}

// Shows on which NUMA nodes the shared pages landed (only mappings shared between processes).
void dump_numa_maps()
{
  std::ifstream maps("/proc/self/numa_maps");
  std::string line;
  std::cout << "--- numa_maps of rank 0, shared mappings:\n";
  while (std::getline(maps, line))
  {
    if (line.find("mapmax=") != std::string::npos && line.find(".so") == std::string::npos)
    {
      std::cout << line << "\n";
    }
  }
  std::cout << std::flush;
}

}  // namespace

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank = 0;
  int nprocs = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  const Options options = parse_options(argc, argv);

  std::vector<Algorithm> algorithms;
  algorithms.push_back({Kind::MpiAllgather, 0});
  for (int k : options.radices)
  {
    if (k >= 2 && k <= options.b) algorithms.push_back({Kind::RadixBatch, k});
  }
  algorithms.push_back({Kind::NodeShm, 0});
  if (options.view) algorithms.push_back({Kind::NodeView, 0});

  std::ofstream csv;
  if (rank == 0)
  {
    csv.open(options.out);
    csv << "algorithm,k,b,nodes,ppn,nprocs,send_count,rep,time_barrier,time_max,"
           "s1_max,s2_max,s3_max,s1_avg,s2_avg,s3_avg,is_correct,read_max\n";
    csv.precision(9);
  }

  const double start = MPI_Wtime();
  bool stopped = false;
  for (int count : options.counts)
  {
    std::vector<int> send(count);
    std::vector<int> recv(static_cast<size_t>(nprocs) * count);
    std::iota(send.begin(), send.end(), rank * count);

    for (const Algorithm& algorithm : algorithms)
    {
      int stop = 0;
      if (rank == 0 && options.max_seconds > 0)
      {
        stop = MPI_Wtime() - start > options.max_seconds;
      }
      MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if (stop)
      {
        if (rank == 0)
        {
          std::cout << "TIME_GUARD stopped before " << name_of(algorithm.kind) << " k=" << algorithm.k
                    << " count=" << count << std::endl;
        }
        stopped = true;
        break;
      }
      run_algorithm(algorithm, options, count, send, recv, csv, rank, nprocs);
      if (rank == 0)
      {
        std::cout << "[" << static_cast<int>(MPI_Wtime() - start) << " s] count=" << count << " "
                  << name_of(algorithm.kind) << " k=" << algorithm.k << std::endl;
      }
    }
    if (stopped) break;
  }

  if (options.diag && rank == 0) dump_numa_maps();
  if (rank == 0) csv.close();

  Allgather_radix_batch_release_cached();
  Allgather_shm_release_cached();
  MPI_Finalize();
  return 0;
}
