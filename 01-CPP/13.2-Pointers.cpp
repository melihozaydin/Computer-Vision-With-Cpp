#include <iostream>
#include <memory>

/*
This script covers:

Dynamic Memory Allocation: Demonstrates how to allocate and deallocate memory dynamically using new and delete.
Smart Pointers: Shows how to use std::unique_ptr and std::shared_ptr for automatic memory management.
Multi-Level Pointers: Demonstrates pointers to pointers (multi-level pointers).
Custom Deleter with Smart Pointers: Shows how to use a custom deleter with std::unique_ptr.
*/

// Function to demonstrate dynamic memory allocation
void dynamicMemoryAllocation() {
    int* ptr = new int(42); // Allocate memory on the heap
    std::cout << "Dynamically allocated value: " << *ptr << "\n";
    delete ptr; // Free the allocated memory

    int* arr = new int[5]{1, 2, 3, 4, 5}; // Allocate array on the heap
    std::cout << "Dynamically allocated array: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << "\n";
    delete[] arr; // Free the allocated array
}

// Function to demonstrate smart pointers
void smartPointers() {
    std::unique_ptr<int> uniquePtr = std::make_unique<int>(42); // Unique pointer
    std::cout << "Unique pointer value: " << *uniquePtr << "\n";

    std::shared_ptr<int> sharedPtr1 = std::make_shared<int>(42); // Shared pointer
    std::shared_ptr<int> sharedPtr2 = sharedPtr1; // Shared ownership
    std::cout << "Shared pointer value: " << *sharedPtr1 << "\n";
    std::cout << "Shared pointer use count: " << sharedPtr1.use_count() << "\n";
}

// Function to demonstrate multi-level pointers
void multiLevelPointers() {
    int var = 42;
    int* ptr = &var; // Pointer to int
    int** ptrToPtr = &ptr; // Pointer to pointer to int

    std::cout << "Value of var: " << var << "\n";
    std::cout << "Value pointed to by ptr: " << *ptr << "\n";
    std::cout << "Value pointed to by ptrToPtr: " << **ptrToPtr << "\n";
}

// Class to demonstrate custom deleter with smart pointers
class CustomDeleter {
public:
    void operator()(int* ptr) const {
        std::cout << "Custom deleter called for pointer: " << ptr << "\n";
        delete ptr;
    }
};

void customDeleterWithSmartPointers() {
    std::unique_ptr<int, CustomDeleter> uniquePtr(new int(42), CustomDeleter());
    std::cout << "Unique pointer with custom deleter value: " << *uniquePtr << "\n";
}

int main() {
    std::cout << "Dynamic Memory Allocation:\n";
    dynamicMemoryAllocation();

    std::cout << "\nSmart Pointers:\n";
    smartPointers();

    std::cout << "\nMulti-Level Pointers:\n";
    multiLevelPointers();

    std::cout << "\nCustom Deleter with Smart Pointers:\n";
    customDeleterWithSmartPointers();

    return 0;
}