#include <iostream>
#include "add.h"
#include "multiply.h"

int main() {
    int a = 5;
    int b = 3;

    std::cout << "Addition: " << add(a, b) << "\n";
    std::cout << "Multiplication: " << multiply(a, b) << "\n";

    return 0;
}