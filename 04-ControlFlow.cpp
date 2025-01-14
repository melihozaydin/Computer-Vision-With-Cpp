#include <iostream>


/*
It is a common misconception that main is always the first function that executes.

Global variables are initialized prior to the execution of main. 
If the initializer for such a variable invokes a function, 
then that function will execute prior to main.
*/

/*
-- Add in For each statements
*/

int main() {
    // ***** ControlFlow *****
    std::cout << "\n ***** ControlFlow ***** \n" << std::endl;

    //Control flow examples in cpp
    // Control flow is the order in which individual statements, instructions or function calls of an imperative program are executed or evaluated.

    // Sequence
    // Sequence is the default control flow in C++.


    // Conditional Statements
    if (true) {
        std::cout << "This is true" << std::endl;
    }

    if (false) {
        std::cout << "This is false" << std::endl;
    }

    // if-else
    if (true) {
        std::cout << "This is true" << std::endl;
    }
    else {
        std::cout << "This is false" << std::endl;
    }

    // if-else-if
    if (true) {
        std::cout << "This is true" << std::endl;
    }
    else if (false) {
        std::cout << "This is false" << std::endl;
    }
    else {
        std::cout << "This is the else" << std::endl;
    }

    // Switch statement
    int n = 2;
    switch (n) {
    case 1:
        std::cout << "Case 1" << std::endl;
        break;
    case 2:
        std::cout << "Case 2" << std::endl;
        break;
    case 3:
        std::cout << "Case 3" << std::endl;
        break;
    }

    // Iteration
    // Iteration is the repetition of a process in order to generate a sequence of outcomes.
    // For loops
    for (int i = 0; i < 5; i++) {
        std::cout << i << " ";
    }

    
    int numbers[] = {1,2,3,4,5} ;

    // Standart for loop for iteration over array
    for (int i=0; i< sizeof(numbers)/sizeof(int); i++){
        std::cout << "Number at index: " << i << " - " << numbers[i];
    }

    // For-each 
    for (int number : numbers){
        std::cout << "Number: " << number;
    }

    // While loops
    int i = 0;
    while (i < 5) {
        std::cout << i << " ";
        i++;
    }

    // Do-While loops
    int i = 0;
    do {
        std::cout << i << " ";
        i++;
    } while (i < 5);

    // Jump Statements
    // Jump statements are used to transfer control to another part of the program.
    // Break statement
    for (int j = 0; j < 10; j++) {
        if (j == 5) {
            break;
        }
        std::cout << j << " ";
    }
    std::cout << std::endl;

    // Continue statement
    for (int j = 0; j < 10; j++) {
        if (j == 5) {
            continue;
        }
        std::cout << j << " ";
    }
    std::cout << std::endl;

    // Goto statement
    int k = 0;
    goto label;
    k = 1; // This line will be skipped
label:
    std::cout << "Goto statement executed, k = " << k << std::endl;


    
}