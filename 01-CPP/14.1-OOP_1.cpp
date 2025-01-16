#include<iostream>
#include <string_view>

/*
In procedural programming, the functions and the data those functions operate on are separate entities.
This leads to code that looks like this:

eat(you, apple);

In Object Oriented programming
the focus is on creating program-defined data types that contain both properties and a set of well-defined behaviors. The term “object” in OOP refers to the objects that we can instantiate from such types.

This leads to code that looks more like this:

you.eat(apple);

Classes are objects that can store both data and functions
This makes it clearer: 
who the subject is (you), 
what behavior is being invoked (eat()), and 
what objects are accessories to that behavior (apple).
*/


//procedural programming style that prints the name and number of legs of an animal
/*
consider what happens when we want to update this program so that our animal is now a snake. 

To add a snake to our code, we’d need to modify AnimalType, numLegs(), animalName(). 

If this were a larger codebase, 
we’d also need to update any other function that uses AnimalType -- if AnimalType was used in a lot of places, 
that could be a lot of code that needs to get touched (and potentially broken).
*/
enum AnimalType
{
    cat,
    dog,
    chicken,
};

constexpr std::string_view animalName(AnimalType type)
{
    switch (type)
    {
    case cat: return "cat";
    case dog: return "dog";
    case chicken: return "chicken";
    default:  return "";
    }
}

constexpr int numLegs(AnimalType type)
{
    switch (type)
    {
    case cat: return 4;
    case dog: return 4;
    case chicken: return 2;
    default:  return 0;
    }
}


// Same program with OOP approach
/*
Now consider the case where we want to update our animal to a snake. 
All we have to do is create a Snake type and use it instead of Cat. 
Very little existing code needs to be changed, 
which means much less risk of breaking something that already works.
*/
struct Cat
{
    std::string_view name{ "cat" };
    int numLegs{ 4 };
};

struct Dog
{
    std::string_view name{ "dog" };
    int numLegs{ 4 };
};

struct Chicken
{
    std::string_view name{ "chicken" };
    int numLegs{ 2 };
};


/*
The term “object”

Note that the term “object” is overloaded a bit, 
and this causes some amount of confusion. 

- In traditional programming, 
-- an object is a piece of memory to store values. And that’s it. 

- In object-oriented programming, 
-- an “object” implies that it is both an object in the traditional programming sense, 
-- and that it combines both properties and behaviors. 

We favor the traditional meaning of the term object, 
and prefer the term “class object” when specifically referring to OOP objects.
*/

int main() {

    // procedural approach
    constexpr AnimalType animal_procedural{ cat };
    std::cout << "A " << animalName(animal_procedural) << " has " << numLegs(animal_procedural) << " legs\n";

    // OOP approach
    constexpr Cat animal_OOP;
    std::cout << "a " << animal_OOP.name << " has " << animal_OOP.numLegs << " legs\n";

    return 0;
}