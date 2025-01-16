#include <iostream>
#include <string>

/*
This script covers:

Basic Struct Definition: The Point struct with public data members and a member function.
Struct with Constructor and Member Functions: The Rectangle struct with a constructor and member functions.
Struct with Encapsulation: The Circle struct with private data members and public member functions, demonstrating encapsulation similar to a class.
*/

// Basic struct definition
struct Point {
    int x;
    int y;

    // Member function to display the point
    void display() const {
        std::cout << "Point(" << x << ", " << y << ")\n";
    }
};

// Struct with constructor and member functions
struct Rectangle {
    int width;
    int height;

    // Constructor
    Rectangle(int w, int h) : width(w), height(h) {}

    // Member function to calculate area
    int area() const {
        return width * height;
    }

    // Member function to display the rectangle
    void display() const {
        std::cout << "Rectangle(width: " << width << ", height: " << height << ")\n";
    }
};

// Struct with encapsulation (similar to a class)
struct Circle {
private:
    double radius;

public:
    // Constructor
    Circle(double r) : radius(r) {}

    // Member function to calculate area
    double area() const {
        return 3.14159 * radius * radius;
    }

    // Member function to display the circle
    void display() const {
        std::cout << "Circle(radius: " << radius << ")\n";
    }
};

int main() {
    // Creating and using a Point struct
    Point p = {3, 4};
    p.display();

    // Creating and using a Rectangle struct
    Rectangle rect(5, 10);
    rect.display();
    std::cout << "Rectangle area: " << rect.area() << "\n";

    // Creating and using a Circle struct
    Circle circle(7.5);
    circle.display();
    std::cout << "Circle area: " << circle.area() << "\n";

    return 0;
}