#include <iostream>
#include <torch/torch.h>

int main() {
  #ifdef DEBUG_MODE
  std::cout << "DEBUG_MODE ACTIVATED\n";
  #endif

  return 0;
}