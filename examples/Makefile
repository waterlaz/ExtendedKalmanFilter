CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra $(shell pkgconf --cflags eigen3) -I../

EXAMPLE_MDS := $(wildcard *.md)
EXAMPLE_SRCS := $(patsubst %.md,%.cpp,$(EXAMPLE_MDS))
EXAMPLE_BINS := $(patsubst %.md,build/%,$(EXAMPLE_MDS))

PROJECT_SRCS := $(wildcard ../*.hpp ../*.cpp)

.PHONY: all clean

all: $(EXAMPLE_SRCS) $(EXAMPLE_BINS)

build:
	mkdir -p build

%.cpp: %.md
	sed -n '/^```/,/^```/ { /^```/d; p; }' $< > $@

build/%: %.cpp $(PROJECT_SRCS) | build
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf build $(EXAMPLE_SRCS)
