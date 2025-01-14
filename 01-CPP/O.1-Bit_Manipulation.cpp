#include<iostream>
#include <bitset> // for std::bitset

int main() {
    //***** Manipulating bits via std::bitset
    /*
    76543210  Bit position
    00000101  Bit sequence
    */

    
    int x{7};// assign x the value 7 (probably uses 32 bits of storage)
    std::cout << x; // print the value 5

    std::bitset<8> bits{ 0b0000'0101 }; // we need 8 bits, start with bit pattern 0000 0101
    bits.set(3);   // set bit position 3 to 1 (now we have 0000 1101)
    bits.flip(4);  // flip bit 4 (now we have 0001 1101)
    bits.reset(4); // set bit 4 back to 0 (now we have 0000 1101)

    std::cout << "All the bits: " << bits<< '\n';
    std::cout << "Bit 3 has value: " << bits.test(3) << '\n';
    std::cout << "Bit 4 has value: " << bits.test(4) << '\n';


}