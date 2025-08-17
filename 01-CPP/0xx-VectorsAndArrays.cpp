#include <iostream>
#include <vector>

int main() {
    // --- Vectors and Arrays ---
    // Vectors are a type of container that can store multiple values of the same type
    // Arrays are a fixed-size sequence of elements of the same type
    // Vectors are more flexible than arrays
    // Vectors can be resized dynamically
    // Arrays have a fixed size that cannot be changed
    // Vectors are part of the C++ Standard Library
    // Arrays are a built-in feature of C++
    // Vectors are preferred over arrays in most cases
    
    // Declare and initialize a vector
    std::vector<int> vec = {1, 2, 3, 4, 5};
    // Print the vector
    for (size_t i = 0; i < vec.size(); i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;

    // Declare and initialize an array
    int arr[5] = {1, 2, 3, 4, 5};
    // Print the array
    for (int i = 0; i < 5; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    // Print the type of the vector and the array
    std::cout << "Type of vector vec: " << typeid(vec).name() << std::endl;
    std::cout << "Type of array arr: " << typeid(arr).name() << std::endl;

    // 2 dimensional array
    int arr2D[2][3] = {{1, 2, 3}, {4, 5, 6}};
    // Print the 2D array
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << arr2D[i][j] << " ";
        }
        std::cout << std::endl;
    }


    // --- Pointers ---
    // Pointers are variables that store the memory address of another variable
    // Pointers are used to store the address of variables
    // Pointers are used to access the memory location of variables
    // Pointers are used to pass variables by reference
    // Pointers are used to allocate memory dynamically
    // Pointers are used to create complex data structures like linked lists, trees, etc.
    // Pointers are used to optimize memory usage and increase performance
    
    // Declare a pointer variable
    int* ptr;
    // Assign the address of a variable to the pointer
    int x = 10;
    ptr = &x;
    // Print the value of the variable using the pointer
    std::cout << "Value of x: " << *ptr << std::endl;

    // Print the memory address of the variable
    std::cout << "Memory address of x: " << ptr << std::endl;

    // Print the memory address of the pointer
    std::cout << "Memory address of ptr: " << &ptr << std::endl;

    // Print the type of the pointer
    std::cout << "Type of pointer ptr: " << typeid(ptr).name() << std::endl;

    return 0;

}