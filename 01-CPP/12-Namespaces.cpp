#include <iostream>
#include <cmath>

/*
Namespaces provide a method for preventing name conflicts in large projects.

Basic Namespace: The Math namespace contains mathematical functions and constants.
Another Namespace: The Geometry namespace contains functions related to geometry, using functions from the Math namespace.
Nested Namespace: The Outer::Inner namespace demonstrates nested namespaces.
Namespace Aliasing: The Geo alias for the Geometry namespace.
Anonymous Namespace: The anonymous namespace for internal linkage, ensuring the function is only accessible within the current translation unit.
Using Keyword: The using keyword to bring specific names or all names from a namespace into the current scope
*/

// Define a namespace for mathematical functions
namespace Math {
    const double PI = 3.14159;

    double square(double x) {
        return x * x;
    }

    double cube(double x) {
        return x * x * x;
    }

    // Function to calculate the hypotenuse of a right triangle
    double hypotenuse(double a, double b) {
        return std::sqrt(square(a) + square(b));
    }
}

// Define another namespace for geometry functions
namespace Geometry {
    double areaOfCircle(double radius) {
        return Math::PI * Math::square(radius);
    }

    double volumeOfSphere(double radius) {
        return (4.0 / 3.0) * Math::PI * Math::cube(radius);
    }

    // Function to calculate the perimeter of a rectangle
    double perimeterOfRectangle(double length, double width) {
        return 2 * (length + width);
    }
}

// Define a nested namespace
namespace Outer {
    namespace Inner {
        void display() {
            std::cout << "Inside nested namespace Outer::Inner\n";
        }
    }
}

// Alias for a namespace
namespace Geo = Geometry;

// Anonymous namespace for internal linkage
namespace {
    void internalFunction() {
        std::cout << "This function is in an anonymous namespace and has internal linkage.\n";
    }
}

int main() {
    // Using the Math namespace
    std::cout << "Square of 3: " << Math::square(3) << "\n";
    std::cout << "Cube of 3: " << Math::cube(3) << "\n";
    std::cout << "Hypotenuse of a right triangle with sides 3 and 4: " << Math::hypotenuse(3, 4) << "\n";

    // Using the Geometry namespace
    std::cout << "Area of circle with radius 5: " << Geometry::areaOfCircle(5) << "\n";
    std::cout << "Volume of sphere with radius 5: " << Geometry::volumeOfSphere(5) << "\n";
    std::cout << "Perimeter of rectangle with length 4 and width 6: " << Geometry::perimeterOfRectangle(4, 6) << "\n";

    // Using the nested namespace
    Outer::Inner::display();

    // Using the alias for Geometry namespace
    std::cout << "Using alias - Area of circle with radius 7: " << Geo::areaOfCircle(7) << "\n";

    // Using the anonymous namespace
    internalFunction();

    // Using the 'using' keyword to bring specific names into scope
    using Math::PI;
    using Math::square;
    std::cout << "Using 'using' keyword - PI: " << PI << "\n";
    std::cout << "Using 'using' keyword - Square of 4: " << square(4) << "\n";

    // Using the 'using namespace' directive to bring all names from a namespace into scope
    using namespace Geometry;
    std::cout << "Using 'using namespace' directive - Area of circle with radius 8: " << areaOfCircle(8) << "\n";
    std::cout << "Using 'using namespace' directive - Volume of sphere with radius 8: " << volumeOfSphere(8) << "\n";

    return 0;
}