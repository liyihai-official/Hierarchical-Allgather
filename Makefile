CC       := mpicc
CXX      := mpicxx
CFLAGS   := -O3 -std=c17
CXXFLAGS := -O3 -std=c++17

TARGET := main
OBJS   := Allgather_radix_batch.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): main.cc $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Comparison benchmark; "make bench_shm". The timed object keeps "main" unchanged.
bench_shm: bench_shm.cc Allgather_radix_batch_timed.o Allgather_shm.o
	$(CXX) $(CXXFLAGS) -DALLGATHER_STAGE_TIMING $^ -o $@

Allgather_radix_batch_timed.o: Allgather_radix_batch.c Allgather_radix_batch.h
	$(CC) $(CFLAGS) -DALLGATHER_STAGE_TIMING -c $< -o $@

Allgather_shm.o: Allgather_shm.c Allgather_shm.h

clean:
	rm -f $(TARGET) $(OBJS) bench_shm Allgather_radix_batch_timed.o Allgather_shm.o