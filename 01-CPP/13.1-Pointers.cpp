#include <iostream>

/*
This script covers:

Basic Pointer Usage: Demonstrates how to declare and use pointers.
Pointer Arithmetic: Shows how to perform arithmetic operations on pointers.
Pointers to Functions: Demonstrates how to declare and use pointers to functions.
Pointers to Members: Shows how to declare and use pointers to member variables and member functions.

Pointers are variables that store the memory address of another variable.
They are used to indirectly access the value of a variable or to dynamically allocate memory.

*/

// Function to demonstrate basic pointer usage
void basicPointerUsage() {
    int var = 42;
    int* ptr = &var; // Pointer to var

    std::cout << "Value of var: " << var << "\n";
    std::cout << "Address of var: " << &var << "\n";
    std::cout << "Value of ptr: " << ptr << "\n";
    std::cout << "Value pointed to by ptr: " << *ptr << "\n";
}

// Function to demonstrate pointer arithmetic
void pointerArithmetic() {
    int arr[5] = {1, 2, 3, 4, 5};
    int* ptr = arr; // Pointer to the first element of the array

    std::cout << "Array elements using pointer arithmetic:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << *(ptr + i) << " ";
    }
    std::cout << "\n";
}

// Function to demonstrate pointers to functions
void functionToPointTo(int x) {
    std::cout << "Function called with value: " << x << "\n";
}

void pointersToFunctions() {
    void (*funcPtr)(int) = functionToPointTo; // Pointer to function
    funcPtr(42); // Calling the function through the pointer
}

// Class to demonstrate pointers to members
class MyClass {
public:
    int memberVar;

    MyClass(int val) : memberVar(val) {}

    void memberFunction() {
        std::cout << "Member function called, memberVar: " << memberVar << "\n";
    }
};

void pointersToMembers() {
    MyClass obj(42);
    int MyClass::*ptrToMember = &MyClass::memberVar; // Pointer to member variable
    void (MyClass::*ptrToMemberFunc)() = &MyClass::memberFunction; // Pointer to member function

    std::cout << "Value of memberVar through pointer: " << obj.*ptrToMember << "\n";
    (obj.*ptrToMemberFunc)(); // Calling the member function through the pointer
}

int main() {
    std::cout << "Basic Pointer Usage:\n";
    basicPointerUsage();

    std::cout << "\nPointer Arithmetic:\n";
    pointerArithmetic();

    std::cout << "\nPointers to Functions:\n";
    pointersToFunctions();

    std::cout << "\nPointers to Members:\n";
    pointersToMembers();

    return 0;
}