.PHONY: all clean

all: example_cc example_rs

clean:
	-rm *.o
	-rm example

## Common

coverage.o:
	clang -g -c coverage.cc -o coverage.o
## C++

example.o:
	clang -g -fsanitize-coverage=trace-pc-guard example.c -c -o example.o

example: coverage.o example.o
	clang++ coverage.o example.o -fsanitize=address -o example