#include <iostream>

int main() {

    // ***** Pointers *****
    std::cout << "\n ******* Pointers ******* \n" << std::endl;

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

    // ***** Pointer functions *****
    std::cout << "\n ******* Pointer Functions ******* \n" << std::endl;

    // Declare and initialize an array
    int arr3[5] = {1, 2, 3, 4, 5};

    // Declare a pointer to the array
    int* ptr2 = arr3;

    // Print the array using the pointer
    for (int i = 0; i < 5; i++) {
        std::cout << *(ptr2 + i) << " ";
    }
    std::cout << std::endl;


    // Pointers are used to allocate memory dynamically
    // Dynamic memory allocation is the process of allocating memory at runtime
    // Dynamic memory allocation is done using the new operator in C++
    // Dynamic memory allocation is used when the size of the data structure is not known at compile time

    // Allocate memory for an integer
    int* ptr3 = new int;
    // Assign a value to the dynamically allocated memory
    *ptr3 = 20;
    // Print the value of the dynamically allocated memory
    std::cout << "Value of dynamically allocated memory: " << *ptr3 << std::endl;
    // Deallocate the dynamically allocated memory
    delete ptr3;

    // Pointers are used to create complex data structures like linked lists, trees, etc.
    // Linked lists are a data structure that consists of nodes linked together
    // Linked lists are used to store data in a non-contiguous manner

    // Declare a linked list node
    struct Node {
        int data;
        Node* next;
    };

    // Create a linked list
    Node* head = new Node;
    head->data = 1;
    head->next = nullptr;

    Node* second = new Node;
    second->data = 2;
    second->next = nullptr;
    head->next = second;

    Node* third = new Node;
    third->data = 3;
    third->next = nullptr;
    second->next = third;

    // Print the linked list
    Node* current = head;
    while (current != nullptr) {
        std::cout << current->data << " ";
        current = current->next;
    }
    std::cout << std::endl;
}
