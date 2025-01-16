#include <iostream>
#include <memory>
#include <vector>

// A simple class to demonstrate RAII
class Resource {
public:
    Resource() {
        std::cout << "Resource acquired\n";
    }
    ~Resource() {
        std::cout << "Resource destroyed\n";
    }
    void doSomething() {
        std::cout << "Doing something with the resource\n";
    }
};

void stackAllocation() {
    // Stack allocation: memory is automatically managed and freed when the function exits
    int stackVar = 42;
    std::cout << "Stack variable: " << stackVar << "\n";
    // Useful for small, short-lived variables
}

void heapAllocation() {
    // Heap allocation: memory is manually managed and must be freed explicitly
    int* heapVar = new int(42);
    std::cout << "Heap variable: " << *heapVar << "\n";
    delete heapVar; // Manual deallocation to avoid memory leaks
    // Useful for large or dynamically-sized data that needs to persist beyond the scope of a function
}

void smartPointers() {
    // Unique pointer: sole ownership of a dynamically allocated object
    std::unique_ptr<int> uniquePtr = std::make_unique<int>(42);
    std::cout << "Unique pointer: " << *uniquePtr << "\n";
    // Automatically deallocates memory when the pointer goes out of scope

    // Shared pointer: shared ownership of a dynamically allocated object
    std::shared_ptr<int> sharedPtr1 = std::make_shared<int>(42);
    std::shared_ptr<int> sharedPtr2 = sharedPtr1; // Shared ownership
    std::cout << "Shared pointer 1: " << *sharedPtr1 << "\n";
    std::cout << "Shared pointer 2: " << *sharedPtr2 << "\n";
    std::cout << "Shared pointer use count: " << sharedPtr1.use_count() << "\n";
    // Automatically deallocates memory when the last shared pointer goes out of scope
    // Useful for managing resources in a multi-owner scenario
}

void resourceManagement() {
    {
        // RAII: Resource is acquired when the object is created
        Resource res;
        res.doSomething();
        // Resource is automatically released when the object goes out of scope
    }
    // RAII ensures that resources are properly released, even if an exception occurs
    // Useful for managing resources like file handles, network connections, etc.
}

int main() {
    std::cout << "Stack Allocation:\n";
    stackAllocation();

    std::cout << "\nHeap Allocation:\n";
    heapAllocation();

    std::cout << "\nSmart Pointers:\n";
    smartPointers();

    std::cout << "\nResource Management (RAII):\n";
    resourceManagement();

    return 0;
}