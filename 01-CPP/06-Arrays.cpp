#include <iostream>

int main () {
    
    // ***** Arrays *****
    std::cout << "\n ******* Arrays ******* \n" << std::endl;

    // Arrays are a fixed-size sequence of elements of the same type
    // Arrays are used to store multiple values of the same type
    // Arrays have a fixed size that cannot be changed
    // Arrays are a built-in feature of C++
    // Arrays are preferred over vectors when the size is known at compile time

    // Declare and initialize an array
    int arr[5] = {1, 2, 3, 4, 5};

    // Print the array
    for (int i = 0; i < 5; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    // Print the type of the array
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

    // find count of elements in an array
    int arr_size = sizeof(arr) / sizeof(arr[0]);
    std::cout << "Size of array arr: " << arr_size << std::endl;


    // ***** Array Iteration *****
    std::cout << "\n ******* Array Iteration ******* \n" << std::endl;

    //array.size() function, which returns the number of elements in the array
    int arr_size = sizeof(arr) / sizeof(arr[0]);
    std::cout << "Size of array arr: " << arr_size << std::endl;

    //array.begin() function, which returns an iterator pointing to the first element of the array
    int* arr_begin = std::begin(arr);
    std::cout << "Begin iterator of array arr: " << *arr_begin << std::endl;

    // Iterate over the array
    for (int i = 0; i < arr_size; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
    


    // ***** Array functions *****
    std::cout << "\n ******* Array Functions ******* \n" << std::endl;


    // Insert elements into an array
    int arr3[5] = {1, 2, 3, 4, 5};
    // Insert 10 at the beginning of the array
    arr3[0] = 10;
    // Insert 20 at the end of the array
    arr3[5] = 20;
    
    // Insert at the end of the array using the size of the array
    int size = sizeof(arr3) / sizeof(arr3[0]);
    arr3[size] = 10;
    
    // Insert 15 at the 3rd position
    arr3[2] = 15;

    // Print the array
    for (int i = 0; i < 5; i++) {
        std::cout << arr3[i] << " ";
    }
    std::cout << std::endl;

    // Delete elements from an array
    // you cant delete elements from an array
    // you can only overwrite the elements with new values
    // vector is a better option if you need to delete elements from a collection

    int arr4[5] = {1, 2, 3, 4, 5};

    // Delete the element at the beginning of the array by setting it to 0
    for (int i = 0; i < 4; i++) {
        arr4[i] = arr4[i + 1];
    }
    std::cout << "Array after deleting the first element: ";
    for (int i = 0; i < 4; i++) {
        std::cout << arr4[i] << " ";
    }


    // Delete the element at the end of the array using the size of the array
    int size2 = sizeof(arr4) / sizeof(arr4[0]);
    arr4[size2 - 1] = 0;
    std::cout << "Array after deleting the last element: ";
    for (int i = 0; i < 4; i++) {
        std::cout << arr4[i] << " ";
    }


    
    // Examples of array functions
    // Arrays do not have built-in functions like vectors
    // You need to write your own functions to perform operations on arrays

    // Find the maximum element in an array
    int max = arr[0];
    for (int i = 1; i < 5; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    std::cout << "Max element in array arr: " << max << std::endl;

    // Find the minimum element in an array
    int min = arr[0];
    for (int i = 1; i < 5; i++) {
        if (arr[i] < min) {
            min = arr[i];
        }
    }

    // Find the sum of elements in an array
    int sum = 0;
    for (int i = 0; i < 5; i++) {
        sum += arr[i];
    }
    std::cout << "Sum of elements in array arr: " << sum << std::endl;


   
    

    return 0;

}