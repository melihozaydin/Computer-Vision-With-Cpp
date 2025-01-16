#include <iostream>

/*
This script covers:

Pointer Arithmetic with Arrays: Demonstrates how to use pointer arithmetic to access elements of an array.
Pointer Comparisons: Shows how to compare pointers.
Pointer Arithmetic with Different Data Types: Demonstrates pointer arithmetic with arrays of different data types (double and char).
*/

// Function to demonstrate pointer arithmetic with arrays
void pointerArithmeticWithArrays() {
    int arr[5] = {10, 20, 30, 40, 50};
    int* ptr = arr; // Pointer to the first element of the array

    std::cout << "Array elements using pointer arithmetic:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "Element " << i << ": " << *(ptr + i) << "\n";
    }
}

// Function to demonstrate pointer comparisons
void pointerComparisons() {
    int arr[5] = {10, 20, 30, 40, 50};
    int* ptr1 = arr; // Pointer to the first element
    int* ptr2 = arr + 4; // Pointer to the last element

    std::cout << "Pointer comparisons:\n";
    std::cout << "ptr1 points to: " << *ptr1 << "\n";
    std::cout << "ptr2 points to: " << *ptr2 << "\n";
    std::cout << "ptr1 < ptr2: " << (ptr1 < ptr2) << "\n";
    std::cout << "ptr1 == ptr2: " << (ptr1 == ptr2) << "\n";
    std::cout << "ptr1 > ptr2: " << (ptr1 > ptr2) << "\n";
}

// Function to demonstrate pointer arithmetic with different data types
void pointerArithmeticWithDifferentTypes() {
    double arr[5] = {1.1, 2.2, 3.3, 4.4, 5.5};
    double* ptr = arr; // Pointer to the first element of the array

    std::cout << "Array elements using pointer arithmetic with double:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "Element " << i << ": " << *(ptr + i) << "\n";
    }

    char charArr[5] = {'a', 'b', 'c', 'd', 'e'};
    char* charPtr = charArr; // Pointer to the first element of the char array

    std::cout << "Array elements using pointer arithmetic with char:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "Element " << i << ": " << *(charPtr + i) << "\n";
    }
}

int main() {
    std::cout << "Pointer Arithmetic with Arrays:\n";
    pointerArithmeticWithArrays();

    std::cout << "\nPointer Comparisons:\n";
    pointerComparisons();

    std::cout << "\nPointer Arithmetic with Different Data Types:\n";
    pointerArithmeticWithDifferentTypes();

    return 0;
}