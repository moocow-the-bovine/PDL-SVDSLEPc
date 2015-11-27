#include <stdio.h>

int main(int argc, const char **argv) {
  int _argc = 2;
  const char *_argv[2] = {"foo","bar"};
  int i;

  for (i=0; i < _argc; ++i) {
    printf("LOCAL: argv[%d/%d] = %s\n", i,_argc,_argv[i]);
  }

  for (i=0; i < argc; ++i) {
    printf("CMDLINE: argv[%d/%d] = %s\n", i,argc,argv[i]);
  }

  return 0;
}
