CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -I/usr/include/eigen3

TEST_SRCS := $(wildcard *.cpp)
PROJECT_SRCS := $(wildcard ../*.hpp ../*.cpp)
TEST_BINS := $(patsubst %.cpp,build/%,$(TEST_SRCS))

.PHONY: all test clean

all: $(TEST_BINS) test

build:
	mkdir -p build

build/%: %.cpp $(PROJECT_SRCS) | build
	$(CXX) $(CXXFLAGS) $< -o $@

test: $(TEST_BINS)
	@for t in $(TEST_BINS); do \
		echo "Running $$t"; \
		./$$t || exit $$?; \
	done

clean:
	rm -rf build
