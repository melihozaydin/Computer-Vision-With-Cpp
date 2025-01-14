#include<iostream>

/*
Variables defined inside the body of a function are called local variables (as opposed to global variables
Function parameters are also generally considered to be local variables, and we will include them as such:
*/
int add(int x, int y)// function parameters x and y are local variables
{
    int z{ x + y }; // z is a local variable
    return z;
}

void doSomething()
{
    std::cout << "Hello!\n";
}

int main() {
    // ***** Local Scope *****


    // ***** Variable lifetime
    // variable definition such as int x; causes the variable to be instantiated when this statement is executed. 
    //int x;


    /*
    Function parameters are created and initialized when the function is entered, 
    and variables within the function body are created and initialized at the point of definition.  
    */

    int x{ 0 };    // x's lifetime begins here
    doSomething(); // x is still alive during this function call

    // To be continued.

    return 0;
} // x's lifetime ends here