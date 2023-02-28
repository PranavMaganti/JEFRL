// example.cc
#include <stdio.h>

void foo() {}

int main(int argc, char **argv) {
  if (argc > 1) {
    foo();
  } else {
    printf("Hello, world!\n");
  }
}