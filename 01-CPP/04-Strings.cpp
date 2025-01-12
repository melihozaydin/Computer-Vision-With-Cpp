#include <iostream>
#include <string>
/*
need fmt library
install with:
vcpkg install fmt
or
sudo apt-get install libfmt-dev
*/
//#include <fmt/core.h>


int main() {
    // ******* Strings ********
    std::cout << "\n ******* Strings ******* \n" << std::endl;
    // Strings are a sequence of characters

    // Declare and initialize a string
    std::string str = "Hello, World!";
    // Print the string
    std::cout << str << std::endl;

    // Get the length of the string
    // str.length() and str.size() are equivalent
    std::cout << "Length of string: " << str.length() << std::endl;
    std::cout << "Size of string: " << str.size() << std::endl;

    // Access individual characters of the string
    std::cout << "First character: " << str[0] << std::endl;

    // Modify individual characters of the string
    str[7] = 'w';
    std::cout << "Modified string: " << str << std::endl;

    // Concatenate strings
    std::string str1 = "Hello, ";
    std::string str2 = "World!";
    std::string str3 = str1 + str2;

    std::cout << "Concatenated string: " << str3 << std::endl;

    // Compare strings
    std::string str4 = "Hello, World!";
    
    std::cout << "Strings are equal: " << (str3 == str4) << std::endl;
    std::cout << "Strings are not equal: " << ("Random string" == str4) << std::endl;

    // Extract substring from a string
    // substr() function takes two arguments: the starting position and the length of the substring
    std::string str6 = "Hello, World!";
    std::string sub1 = str6.substr(7, 5);
    std::cout << "Extracted Substring: " << sub1 << std::endl;


    // ***** Convert Data Types *****
    std::cout << "\n ******* Convert Data Types ******* \n" << std::endl;
    // Convert string to integer
    //stoi() function converts a string to an integer
    //stoi()
    std::string str7 = "12345";
    int num = std::stoi(str7);
    std::cout << "String to Number: " << num << std::endl;

    // Convert string to float
    //stof() function converts a string to a float
    std::string str8 = "3.14159";
    float num1 = std::stof(str8);
    std::cout << "String to Float: " << num1 << std::endl;

    // Convert string to double
    //stod() function converts a string to a double
    double num2 = std::stod(str8);
    std::cout << "String to Double: " << num2 << std::endl;

    // Convert integer to string
    // to_string can accept any numeric type including int, long, long long, unsigned, etc.
    int num3 = 54321;
    std::string str9 = std::to_string(num3);
    std::cout << "Number to String: " << str9 << std::endl;

    // Print the type of the string
    std::cout << "Type of string str: " << typeid(str).name() << std::endl;

    // ***** String Iteration *****
    std::cout << "\n ******* String Iteration ******* \n" << std::endl;

    // Iterate over the string
    for (int i = 0; i < str.length(); i++) {
        std::cout << str[i] << " ";
    }
    std::cout << std::endl;

    // Iterate over the string using iterators
    for (std::string::iterator it = str.begin(); it != str.end(); it++) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    // Iterate over the string using range-based for loop
    for (char c : str) {
        std::cout << c << " ";
    }
    std::cout << std::endl;
    
    // ***** String Functions *****
    std::cout << "\n ******* String Functions ******* \n" << std::endl;

    // Check if the string is empty
    std::cout << "Is Empty string ? (bool): " << str.empty() << std::endl;

    // Append to a string
    std::string str10 = "Hello, ";
    str10.append("World!");
    std::cout << "Appended string: " << str9 << std::endl;

    // Erase characters from a string
    // erase() function takes two arguments:
    // the starting position and 
    // the number of characters to erase
    {
    std::string str11 = "Hello, World!";
    str11.erase(7, 6);
    std::cout << "Erased string: " << str11 << std::endl;
    }
    // Insert characters into a string
    {
    std::string str11 = "Hello, World!";
    str11.insert(7, "Beautiful ");
    std::cout << "Inserted string: " << str11 << std::endl;
    }

    // Replace characters in a string
    {
    std::string str12 = "Hello, World!";
    str12.replace(7, 5, "Universe");
    std::cout << "Replaced string: " << str12 << std::endl;
    }
    
    // Swap two strings
    {
    std::string str13 = "Hello, World!";
    std::string str14 = "Goodbye, World!";
    str13.swap(str14);
    std::cout << "Swapped strings: " << str13 << " " << str14 << std::endl;

    }

    // Clear a string
    std::string str15 = "Hello, World!";
    str15.clear();
    std::cout << "Cleared string: " << str15 << std::endl;

    // Convert string to uppercase
    std::string str22 = "Hello, World!";
    for (char &c : str22) {
        c = toupper(c);
    }
    std::cout << "Uppercase string: " << str22 << std::endl;

    // Convert string to lowercase
    std::string str23 = "Hello, World!";
    for (char &c : str23) {
        c = tolower(c);
    }
    std::cout << "Lowercase string: " << str23 << std::endl;

    
    // ***** Search Functions *****
    std::cout << "\n ***** Search Functions ***** \n" << std::endl;

    // Find substring in a string
    std::string str5 = "Hello, World!";
    std::string sub = "World";
    size_t found = str5.find(sub);

    // Check if the substring is found
    // If the substring is not found, 
    // std::string::find returns a special constant value, std::string::npos.
    if (found != std::string::npos) {
        std::cout << "Substring found at position: " << found << std::endl;
    } else {
        std::cout << "Substring not found" << std::endl;
    }

    // Find the first occurrence of a character in a string
    std::string str16 = "Hello, World!";
    size_t found1 = str16.find_first_of('o');
    if (found1 != std::string::npos) {
        std::cout << "First occurrence of 'o' at position: " << found1 << std::endl;
    } else {
        std::cout << "Character not found" << std::endl;
    }

    // Find the last occurrence of a character in a string
    std::string str17 = "Hello, World!";
    size_t found2 = str17.find_last_of('o');
    if (found2 != std::string::npos) {
        std::cout << "Last occurrence of 'o' at position: " << found2 << std::endl;
    } else {
        std::cout << "Character not found" << std::endl;
    }

    // Find the first occurrence of a substring in a string
    std::string str18 = "Hello, World!";
    size_t found3 = str18.find_first_of("World");
    if (found3 != std::string::npos) {
        std::cout << "First occurrence of 'World' at position: " << found3 << std::endl;
    } else {
        std::cout << "Substring not found" << std::endl;
    }

    // Find the last occurrence of a substring in a string
    std::string str19 = "Hello, World!";

    size_t found4 = str19.find_last_of("World");
    if (found4 != std::string::npos) {
        std::cout << "Last occurrence of 'World' at position: " << found4 << std::endl;
    } else {
        std::cout << "Substring not found" << std::endl;
    }

    // Find the first occurrence of a character not in a string
    std::string str20 = "Hello, World!";
    size_t found5 = str20.find_first_not_of("Hello");

    if (found5 != std::string::npos) {
        std::cout << "First occurrence of character not in 'Hello' at position: " << found5 << std::endl;
    } else {
        std::cout << "Character not found" << std::endl;
    }

    // Find the last occurrence of a character not in a string
    std::string str21 = "Hello, World!";
    size_t found6 = str21.find_last_not_of("World");

    if (found6 != std::string::npos) {
        std::cout << "Last occurrence of character not in 'World' at position: " << found6 << std::endl;
    } else {
        std::cout << "Character not found" << std::endl;
    }

    // Remove whitespace from the beginning and end of a string
    std::string str24 = "   Hello, World!   ";
    str24.erase(0, str24.find_first_not_of(" "));
    str24.erase(str24.find_last_not_of(" ") + 1);

    std::cout << "Trimmed string: " << str24 << std::endl;

    // Remove whitespace from the beginning of a string
    std::string str25 = "   Hello, World!   ";
    str25.erase(0, str25.find_first_not_of(" "));
    std::cout << "Trimmed string: " << str25 << std::endl;

    // Remove whitespace from the end of a string
    std::string str26 = "   Hello, World!   ";
    str26.erase(str26.find_last_not_of(" ") + 1);
    std::cout << "Trimmed string: " << str26 << std::endl;


    // ***** Formatted Strings *****
    std::cout << "\n ******* *******" << std::endl;
    // Formatted string using printf (fmt library)
//    int x = 10;
//    float y = 3.14;
//    std::string str27 = fmt::format("x = {}, y = {}", x, y);
//    std::cout << "Formatted string: " << str27 << std::endl;


    return 0;
}