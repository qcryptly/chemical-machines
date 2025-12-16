// %%
#include <iostream>
#include "./another-file.hpp"
int main() {
  hello v{.x = 10};
  std::cout << "Hello! " << v.x << std::endl;
  return 0;
}
