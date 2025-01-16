#include <iostream>
#include <string>


int main() {
      
    // Declare and initialize variables of different types
    int y = 10;       // Integer variable to store whole numbers
    float f = 3.14; // Float variable to store decimal numbers with less precision
    bool flag = true; // Boolean variable to store true or false values
    char c = 'A';     // Character variable to store single characters
    std::string str = "Hello, world!"; // String variable to store a sequence of characters

    double z = 3.14159; // Double variable to store decimal numbers with more precision
    long x = 1234567890; // Long variable to store large whole numbers
    long long xx = 1234567890123456789; // Long long variable to store very large whole numbers
    short s = 10; // Short variable to store small whole numbers

    // Strings
    /*
    strings are a class that represents a sequence of characters
    Using std::string is generally considered better than using C-style strings
    */
    // const indicates that the value of the variable cannot be changed
    const char* str1 = "Hello, world!"; // C-style string  variable
    //std:string
    std::string str2 = "Hello, world!";

    // Print the values of the variables
    std::cout << "int y: " << y << std::endl;
    std::cout << "float f: " << f << std::endl;
    std::cout << "bool flag: " << flag << std::endl;
    std::cout << "double z: " << z << std::endl;
    std::cout << "char c: " << c << std::endl;
    std::cout << "C-style string str: " << str2 << std::endl;
    std::cout << "std::string str: " << str2 << std::endl;
    std::cout << "long x: " << x << std::endl;
    std::cout << "long long xx: " << xx << std::endl;
    std::cout << "short s: " << s << std::endl;

    std::cout << "\n" << std::endl;
    //******************

    // Print the size of the variables
    // sizeof operator can be used to get the size of variables
    // returns the size of the variable in bytes

    // return datatype of sizeof() function is size_t 
    //(m stands for unsigned long. It is a type that represents the size of objects in bytes)
    std::cout << "Returned data type of sizeof(int) function : " << typeid(sizeof(int)).name() << std::endl;

    int size_int = sizeof(int); // 4 bytes
    int size_float = sizeof(float); // 4 bytes
    int size_bool = sizeof(bool); // 1 byte
    int size_char = sizeof(char); // 1 byte
    int size_double = sizeof(double); // 8 bytes
    int size_long = sizeof(long); // 4 bytes
    int size_long_long = sizeof(long long); // 8 bytes
    int size_short = sizeof(short); // 2 bytes

    std::cout << "sizeof(int): " << sizeof(int) << std::endl;
    std::cout << "sizeof(float): " << sizeof(float) << std::endl;
    std::cout << "sizeof(bool): " << sizeof(bool) << std::endl;
    std::cout << "sizeof(char): " << sizeof(char) << std::endl;
    std::cout << "sizeof(double): " << sizeof(double) << std::endl;
    std::cout << "sizeof(long): " << sizeof(long) << std::endl;
    std::cout << "sizeof(long long): " << sizeof(long long) << std::endl;
    std::cout << "sizeof(short): " << sizeof(short) << std::endl;

    std::cout << "\n" << std::endl;
    //******************

    // Print the type of the variables
    // typeid(x) can be used to get type data of variables
    std::cout << "Type of int y: " << typeid(y).name() << std::endl;
    std::cout << "Type of double z: " << typeid(z).name() << std::endl;
    std::cout << "Type of char c: " << typeid(c).name() << std::endl;
    std::cout << "Type of bool flag: " << typeid(flag).name() << std::endl;

    std::cout << "\n" << std::endl;
    /*  
        Type of C-style string str1: PKc::
        P stands for "pointer".
        K stands for "const".
        c stands for "char".
        So, PKc means "pointer to a constant character" (const char*). 
        This is the type of C-style strings.
    */
    std::cout << "Type of C-style string str1: " << typeid(str1).name() << " <<stands for: pointer to a constant character>>" << std::endl;
    std::cout << "Type of std::string str2: " << typeid(str2).name() << std::endl;

    // what type does typeid() itself return? (PKc like C-style strings)
    std::cout << "Return type of typeid(x).name() return: " << typeid(typeid(x).name()).name() << std::endl;

    // ********* unsigned types *********
    // unsigned types can store 0 and positive values
    // unsigned types have a larger range of positive values compared to signed types
    // unsigned types have the same size as signed types
    // used when negative values are not needed
    unsigned int u = 10; // unsigned int variable to store positive whole numbers
    unsigned long ul = 1000000000;
    unsigned long long ull = 1000000000000000000;
    unsigned short us = 10;

    std::cout << "unsigned int u: " << u << std::endl;
    std::cout << "unsigned long ul: " << ul << std::endl;
    std::cout << "unsigned long long ull: " << ull << std::endl;
    std::cout << "unsigned short us: " << us << std::endl;

    return 0;
}