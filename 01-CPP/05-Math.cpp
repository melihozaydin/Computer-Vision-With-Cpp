#include <iostream>
#include <cmath> // Include the <cmath> header for the sqrt() function

int main() {

    // ***** Math *****
    std::cout << "\n ******* Math ******* \n" << std::endl;

    // Math operations
    int a = 10;
    int b = 20;

    // Addition
    int sum = a + b;
    std::cout << "Sum: " << sum << std::endl;

    // Subtraction
    int diff = a - b;
    std::cout << "Difference: " << diff << std::endl;

    // Multiplication
    int prod = a * b;
    std::cout << "Product: " << prod << std::endl;

    // Division
    int div = a / b;
    std::cout << "Division: " << div << std::endl;

    // Modulus
    int mod = a % b;
    std::cout << "Modulus: " << mod << std::endl;

    // Increment and Decrement
    int x = 10;
    int y = 20;

    // Increment
    x++;
    std::cout << "Incremented x: " << x << std::endl;

    // Decrement
    y--;
    std::cout << "Decremented y: " << y << std::endl;

    // Pre-increment
    int z = 10;
    int pre_inc = ++z;
    std::cout << "Pre-incremented z: " << pre_inc << std::endl;

    // Pre-decrement
    int w = 20;
    int pre_dec = --w;
    std::cout << "Pre-decremented w: " << pre_dec << std::endl;

    // round() function
    double num1 = 3.14;
    double num2 = 3.75;

    // Round a number to the nearest integer
    std::cout << "Round 3.14: " << round(num1) << std::endl;
    std::cout << "Round 3.75: " << round(num2) << std::endl;

    // ceil() function
    // Ceil function rounds a number up to the nearest integer
    std::cout << "Ceil 3.14: " << ceil(num1) << std::endl;

    // floor() function
    // Floor function rounds a number down to the nearest integer
    std::cout << "Floor 3.75: " << floor(num2) << std::endl;

    // abs() function
    // Abs function returns the absolute value of a number
    int num3 = -10;
    std::cout << "Absolute value of -10: " << abs(num3) << std::endl;

    // pow() function
    // Pow function returns the value of a number raised to the power of another number
    int base = 2;
    int exponent = 3;
    std::cout << "2^3: " << pow(base, exponent) << std::endl;

    // sqrt() function
    // Sqrt function returns the square root of a number
    int num4 = 16;

    std::cout << "Square root of 16: " << sqrt(num4) << std::endl;

    // cbrt() function
    // Cbrt function returns the cube root of a number
    int num5 = 27;
    std::cout << "Cube root of 27: " << cbrt(num5) << std::endl;
    std::cout << "Cube root of 15: " << cbrt(15) << std::endl;

    // fmod() function
    // Fmod function returns the remainder of a division operation
    double num6 = 10.5;
    double num7 = 3.5;
    std::cout << "Fmod 10.5 / 3.5: " << fmod(num6, num7) << std::endl;

    // max() function
    // Max function returns the maximum of two numbers
    int num8 = 10;
    int num9 = 20;
    std::cout << "Max of 10 and 20: " << fmax(num8, num9) << std::endl;

    // min() function
    // Min function returns the minimum of two numbers
    int num10 = 10;
    int num11 = 20;
    std::cout << "Min of 10 and 20: " << fmin(num10, num11) << std::endl;

    // ***** Logarithmic Functions *****
    std::cout << "\n ******* Logarithmic Functions ******* \n" << std::endl;
    
    // log() function
    // Log function returns the natural logarithm of a number
    double num12 = 2.71828;
    std::cout << "Log of 2.71828: " << log(num12) << std::endl;

    // log10() function
    // Log10 function returns the base 10 logarithm of a number
    double num13 = 100;
    std::cout << "Log10 of 100: " << log10(num13) << std::endl;

    // log2() function
    // Log2 function returns the base 2 logarithm of a number
    double num14 = 8;
    std::cout << "Log2 of 8: " << log2(num14) << std::endl;

    // log1p() function
    // Log1p function returns the natural logarithm of a number plus 1
    double num19 = 1;
    std::cout << "Log(1 + 1): " << log1p(num19) << std::endl;
    

    // ***** Trigonometric Functions *****
    std::cout << "\n ******* Trigonometric Functions ******* \n" << std::endl;

    // sin() function
    // Sin function returns the sine of an angle
    // The angle must be in radians
    double angle = M_PI / 2; // 90 degrees in radians
    
    // convert degrees to radians
    // angle = angle * M_PI / 180;
    // convert radians to degrees
    // angle = angle * 180 / M_PI;

    std::cout << "Sin of 90 degrees: " << sin(angle) << std::endl;

    // cos() function
    // Cos function returns the cosine of an angle
    std::cout << "Cos of 90 degrees: " << cos(angle) << std::endl;

    // tan() function
    // Tan function returns the tangent of an angle
    std::cout << "Tan of 90 degrees: " << tan(angle) << std::endl;

    // asin() function
    // Asin function returns the arcsine of a number
    double num15 = 1;
    std::cout << "Arcsine of 1: " << asin(num15) << std::endl;

    // acos() function
    // Acos function returns the arccosine of a number
    std::cout << "Arccosine of 1: " << acos(num15) << std::endl;

    // atan() function
    // Atan function returns the arctangent of a number
    std::cout << "Arctangent of 1: " << atan(num15) << std::endl;

    // sinh() function
    // Sinh function returns the hyperbolic sine of a number
    double num16 = 1;
    std::cout << "Hyperbolic sine of 1: " << sinh(num16) << std::endl;

    // cosh() function
    // Cosh function returns the hyperbolic cosine of a number
    std::cout << "Hyperbolic cosine of 1: " << cosh(num16) << std::endl;

    // tanh() function
    // Tanh function returns the hyperbolic tangent of a number
    std::cout << "Hyperbolic tangent of 1: " << tanh(num16) << std::endl;

    // asinh() function
    // Asinh function returns the inverse hyperbolic sine of a number
    double num17 = 1;
    std::cout << "Inverse hyperbolic sine of 1: " << asinh(num17) << std::endl;

    // acosh() function
    // Acosh function returns the inverse hyperbolic cosine of a number
    std::cout << "Inverse hyperbolic cosine of 1: " << acosh(num17) << std::endl;

    // atanh() function
    // Atanh function returns the inverse hyperbolic tangent of a number
    std::cout << "Inverse hyperbolic tangent of 1: " << atanh(num17) << std::endl;

    // exp() function
    // Exp function returns the value of e raised to the power of a number
    double num18 = 1;
    std::cout << "e^1: " << exp(num18) << std::endl;

    // exp2() function
    // Exp2 function returns the value of 2 raised to the power of a number
    std::cout << "2^1: " << exp2(num18) << std::endl;

    // expm1() function
    // Expm1 function returns the value of e raised to the power of a number minus 1
    std::cout << "e^1 - 1: " << expm1(num18) << std::endl;

    // hypot() function
    // Hypot function returns the square root of the sum of the squares of two numbers
    double num20 = 3;
    double num21 = 4;
    std::cout << "Hypotenuse of 3 and 4: " << hypot(num20, num21) << std::endl;

    // trunc() function
    // Trunc function returns the integer part of a number
    double num22 = 3.14;
    std::cout << "Truncate 3.14: " << trunc(num22) << std::endl;

    // **** Random Numbers ****
    std::cout << "\n ******* Random Numbers ******* \n" << std::endl;

    // rand() function
    // Rand function generates a random number
    int random_num = rand();
    std::cout << "Random number: " << random_num << std::endl;

    // srand() function
    // Srand function seeds the random number generator
    srand(0);
    int random_num1 = rand();
    std::cout << "Random number: " << random_num1 << std::endl;

    // Generate random numbers between a range
    int min = 1;
    int max = 10;
    int random_num2 = min + (rand() % (max - min + 1));
    std::cout << "Random number between 1 and 10: " << random_num2 << std::endl;

    //***** Constants *****
    std::cout << "\n ******* Constants ******* \n" << std::endl;

    // M_PI constant
    // M_PI constant represents the value of pi
    std::cout << "Value of pi: " << M_PI << std::endl;

    // M_E constant
    // M_E constant represents the value of e
    std::cout << "Value of e: " << M_E << std::endl;

    // M_LOG2E constant
    // M_LOG2E constant represents the value of log2(e)
    std::cout << "Value of log2(e): " << M_LOG2E << std::endl;

    // M_LOG10E constant
    // M_LOG10E constant represents the value of log10(e)
    std::cout << "Value of log10(e): " << M_LOG10E << std::endl;

    // M_LN2 constant
    // M_LN2 constant represents the value of ln(2)
    std::cout << "Value of ln(2): " << M_LN2 << std::endl;

    // M_LN10 constant
    // M_LN10 constant represents the value of ln(10)
    std::cout << "Value of ln(10): " << M_LN10 << std::endl;

    // M_SQRT2 constant
    // M_SQRT2 constant represents the value of sqrt(2)
    std::cout << "Value of sqrt(2): " << M_SQRT2 << std::endl;

    // M_SQRT1_2 constant
    // M_SQRT1_2 constant represents the value of 1/sqrt(2)
    std::cout << "Value of 1/sqrt(2): " << M_SQRT1_2 << std::endl;

    // M_2_SQRTPI constant
    // M_2_SQRTPI constant represents the value of 2/sqrt(pi)
    std::cout << "Value of 2/sqrt(pi): " << M_2_SQRTPI << std::endl;

    // M_1_PI constant
    // M_1_PI constant represents the value of 1/pi
    std::cout << "Value of 1/pi: " << M_1_PI << std::endl;

    // M_2_PI constant
    // M_2_PI constant represents the value of 2/pi
    std::cout << "Value of 2/pi: " << M_2_PI << std::endl;

    // M_PI_2 constant
    // M_PI_2 constant represents the value of pi/2
    std::cout << "Value of pi/2: " << M_PI_2 << std::endl;

    // M_PI_4 constant
    // M_PI_4 constant represents the value of pi/4
    std::cout << "Value of pi/4: " << M_PI_4 << std::endl;

    return 0;
}