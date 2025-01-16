#include <iostream>
#include <string>

/*
This script covers:
Inheritance: Dog and Cat classes inherit from the Animal base class.
Polymorphism: The makeAnimalSpeak function demonstrates polymorphism by calling the overridden speak method on different derived class objects.
Encapsulation: The name attribute is protected and accessed via a public getter method.
Abstraction: The Animal class is abstract, with a pure virtual function speak.
*/


// Base class (abstract class)
class Animal {
protected:
    std::string name;

public:
    Animal(const std::string& name) : name(name) {}
    virtual ~Animal() {}

    // Pure virtual function (abstract method)
    virtual void speak() const = 0;

    // Encapsulation: getter for name
    std::string getName() const {
        return name;
    }
};

// Derived class inheriting from Animal
class Dog : public Animal {
public:
    Dog(const std::string& name) : Animal(name) {}

    // Overriding the pure virtual function
    void speak() const override {
        std::cout << name << " says: Woof!\n";
    }
};

// Another derived class inheriting from Animal
class Cat : public Animal {
public:
    Cat(const std::string& name) : Animal(name) {}

    // Overriding the pure virtual function
    void speak() const override {
        std::cout << name << " says: Meow!\n";
    }
};

// Function demonstrating polymorphism
void makeAnimalSpeak(const Animal& animal) {
    animal.speak();
}

int main() {
    // Creating objects of derived classes
    Dog dog("Buddy");
    Cat cat("Whiskers");

    // Demonstrating encapsulation
    std::cout << "Dog's name: " << dog.getName() << "\n";
    std::cout << "Cat's name: " << cat.getName() << "\n";

    // Demonstrating polymorphism
    makeAnimalSpeak(dog);
    makeAnimalSpeak(cat);

    return 0;
}