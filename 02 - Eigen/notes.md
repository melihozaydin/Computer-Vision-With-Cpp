https://eigen.tuxfamily.org/dox/GettingStarted.html

# the long tutorial
https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html

## Install eigen
Compiling and running your first program

There is no library to link to. The only thing that you need to keep in mind when compiling the above program is that the compiler must be able to find the Eigen header files. The directory in which you placed Eigen's source code must be in the include path. With GCC you use the -I option to achieve this, so you can compile the program with a command like this:
```bash
g++ -I /path/to/eigen/ my_program.cpp -o my_program 
```

On Linux or Mac OS X, another option is to symlink or copy the Eigen folder into /usr/local/include/. This way, you can compile the program with:
```bash
g++ my_program.cpp -o my_program 
```

# Compile code
```bash
g++ ./02\ -\ Eigen/00-Eigen_Basics.cpp -I /usr/include/eigen3 -o .build/00-Eigen_Basics && ./.build/00-Eigen_Basics
```