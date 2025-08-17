#include <iostream>
#include <vector>

// Function Declaration
// This is the function header (tells the compiler about the existence of the function)
void printVector1D(std::vector<int> vec);

// Function Definition
void printVector1D(std::vector<int> vec) {
    std::cout << "printVector1D: " << std::endl;
    std::cout << "\t";
    for (int i : vec) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

// You can declare and define function in one step
// here vec is the parameter as it is the decleration
void printVector2D(std::vector<std::vector<int>> vec) {
    std::cout << "printVector2D: " << std::endl;
    for (std::vector<int> vec1D : vec) {
        std::cout << "\t";
        printVector1D(vec1D);
    }
}

// Functions can return values
std::vector<int> DoubleVector1D(std::vector<int> vec) {
    std::cout << "vec1D Doubled" << std::endl;
        for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = vec[i] * 2;
    }
    return vec;
}

std::vector<std::vector<int>> dotProduct(std::vector<int>vec1, std::vector<std::vector<int>> vec2) {

    //check if either vector is empty
    if (vec1.empty() || vec2.empty()) {
        std::cout << "One of the vectors is empty." << std::endl;
        // Raise Exception
        throw std::invalid_argument("One of the vectors is empty.");
    }

    // Check if the sizes of the vectors match
    int vec2_row = vec2.size();
    int vec2_col = vec2[0].size();
    int vec1_row = vec1.size();

    if (vec2_col != vec1_row) {
        std::cout << "The sizes of the vectors do not match." << std::endl;
        std::cout << "vec1 size: " << vec1_row << std::endl;
        std::cout << "vec2 size: " << vec2_row << "x" << vec2_col << std::endl;
        
        // Raise Exception
        throw std::invalid_argument("The sizes of the vectors do not match.");
    }

    // Implement dot product of 2D vector and 1D vector
        for (size_t i = 0; i < vec2.size(); i++) {
            for (size_t j = 0; j < vec2[i].size(); j++) {
            vec2[i][j] = vec1[j] * vec2[i][j];
        }
    }

    return vec2;
}


void doSomething(int) // ok: unnamed parameter will not generate warning
{
    std::cout << "doSomething called" << std::endl;
}
    

// ***** Function Overloading *****
// You can have two functions with the same name as long as the have different parameters (declearations)

void doSomething(int val1, int val2) // ok: this is a version of "doSomething" that takes two integers instead of one
{
    std::cout << "doSomething (overloaded) called" << std::endl;
    std::cout << "val1: " << val1 << " val2: " << val2 << std::endl;
}



// Definition of user-defined function main()
int main() {
    std::cout << "Starting main()\n";

    doSomething(1);
    doSomething(1, 2);

    // ***** Functions *****
    std::cout << "***** Functions *****" << std::endl;

    // Function Call
    std::vector<int> vec1D = {1, 2, 3, 4, 5}; 
    printVector1D(vec1D); // Interrupt main() by making a function call to printVector().  main() is the caller.

    // Functions can call functions that call other functions
    // Function Call
    std::vector<std::vector<int>> vec2D = {
                                        {1, 2, 3},
                                        {4, 5, 6},
                                        {7, 8, 9}
                                    };

    // printVector1D(vec2D[0]); // This will not work because printVector1D expects a 1D vector
    printVector2D(vec2D); // printVector2D calls printVector1D which prints each element in each dim.


    // Nested functions are not supported 
    // Illegal: this function is nested inside function main()
    // void foo() {std::cout << "foo!\n";}
    // The proper wayis to define the function outside of main()

    //foo(); // function call to foo() will not work (Compiler error.)


    // **** Function Return ****
    // Functions can return values
    // Function Call
    vec1D = DoubleVector1D(vec1D);
    printVector1D(vec1D);


    // ***** Void Functions *****
    // Functions that do not return a value are called void functions
    // void functions are used when you do not need to return a value
    
    // printVector1D is a void function
    printVector1D(vec1D);

    // trying to get output will raise 'compiler error: error: void value not ignored' as it ought to be'
    // Returning a value from a void function is also a compile error
    //int test = printVector1D(vec1D);

    // *****  Parameters and Arguments *****
    std::cout << "***** Parameters and Arguments *****" << std::endl;
    
    /* 
    A function parameter is a variable used in the header of a function. 
    An argument is a value that is passed from the caller to the function when a function call is made:

    Function parameters work almost identically to variables defined inside the function, 
    but with one difference: 
    -- they are initialized with a value provided by the caller of the function.
    
    Function parameters are defined in the function header 
    by placing them in between the parenthesis after the function name, 
    with multiple parameters being separated by commas.
    */

    // here "vec2D" is the argument it could have been anything 
    // "vec" is the parameter name in the definition above
    printVector2D(vec2D);

    // ***** Multiple Arguments *****
    std::cout << "***** Multiple Arguments *****" << std::endl;

    // Functions can have multiple parameters
    try {
        // dot products of vec1D and vec2D 
        // will raise an error due to size mismatch
        std::vector<std::vector<int>> vec3 = dotProduct(vec1D, vec2D);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    // Correct size
    std::vector<std::vector<int>> vec3 = dotProduct({4,5,6}, vec2D);
    // Print result
    std::cout << "Dot product result: ";
    printVector2D(vec3);

    //Your main function should return the value 0 if the program ran normally.
    // C++ standard only defines the meaning of 3 status codes: 0, EXIT_SUCCESS, and EXIT_FAILURE
    // 0 and EXIT_SUCCESS both mean the program executed successfully. 
    // EXIT_FAILURE means the program did not execute successfully.
    std::cout << "\n *********** Ending main() ***********\n";
    //return 0;
    return EXIT_SUCCESS;

}