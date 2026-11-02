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

clean:
	rm -f $(TARGET) $(OBJS)