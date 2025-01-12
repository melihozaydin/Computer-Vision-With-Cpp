#include <iostream>
#include <string>



int main() {
    // Declare a variable of type int
    int x;
    // Assign a value to the variable
    x = 5;
    // Print the value of the variable
    std::cout << x << std::endl;
  
    /*
    In C++, you cannot declare two variables with the same name in the same scope. 
    Each variable must have a unique name within its scope. 
    If you need to reuse a variable name, you can do so by 
    limiting the scope of the original variable using a block scope 
    or by reassigning the value to the existing variable.
    */

   //block scope
    {
        int x = 10;
        std::cout << "x in block scope: " << x << std::endl;
    }// x goes out of scope here

    //reassigning the value to the existing variable
    std::cout << "x before reassignment: " << x << std::endl;
    x = 15;
    std::cout << "x after reassignment: " << x << std::endl;

    return 0;
}