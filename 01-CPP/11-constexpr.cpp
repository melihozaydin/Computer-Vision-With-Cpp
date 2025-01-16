#include <iostream>

/*
constexpr, which allows for compile-time constant expressions. 
This can be useful for optimizing performance and ensuring certain values are computed at compile time.

This script covers:

constexpr Function: The factorial function is marked as constexpr, allowing it to be evaluated at compile time.
constexpr Variable: The value variable is marked as constexpr, ensuring it is a compile-time constant.
constexpr Class: The ConstExprClass class has a constexpr constructor and member function, allowing objects of this class to be created and used in constexpr contexts.
*/


// constexpr function to calculate the factorial of a number
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : (n * factorial(n - 1));
}

// constexpr variable
constexpr int value = 10;

// constexpr class with a constexpr constructor
class ConstExprClass {
public:
    constexpr ConstExprClass(int x) : value(x) {}
    constexpr int getValue() const { return value; }

private:
    int value;
};

int main() {
    // Using constexpr function
    constexpr int factValue = factorial(5);
    std::cout << "Factorial of 5 is: " << factValue << "\n";

    // Using constexpr variable
    std::cout << "Value is: " << value << "\n";

    // Using constexpr class
    constexpr ConstExprClass obj(42);
    std::cout << "ConstExprClass value is: " << obj.getValue() << "\n";

    return 0;
}