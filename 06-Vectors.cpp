#include <iostream>
#include <vector>

int main() {
    // ***** Vectors *****
    std::cout << "\n ***** Vectors ***** \n" << std::endl;

    // Declare a 1D vector with values
    std::vector<int> vec1D = {1, 2, 3, 4, 5};
    
    // Declare a 2D vector with values
    std::vector<std::vector<int>> vec2D = {
                                            {1, 2, 3}, 
                                            {4, 5, 6}, 
                                            {7, 8, 9}
                                            };

    // Declare a 3D vector with values
    std::vector<std::vector<std::vector<int>>> vec3D = {
                                                        {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}}, 
                                                        {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}}, 
                                                        {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}}
                                                        };
    



    // ***** Vector Functions *****
    std::cout << "\n ***** Vector Functions ***** \n" << std::endl;

    // Get sizes of Vectors
    std::cout << "Size of the 1D vector: " << vec1D.size() << std::endl;
    std::cout << "Size of the 2D vector: " << vec2D.size() << std::endl;
    std::cout << "Size of the 3D vector: " << vec3D.size() << std::endl;

    // Get vector dim (eg. Vec3D(3,3,3))
    std::cout << "Dimensions of vec3D: (" 
              << vec3D.size() << ", " 
              << (vec3D.size() > 0 ? vec3D[0].size() : 0) << ", " 
              << (vec3D.size() > 0 && vec3D[0].size() > 0 ? vec3D[0][0].size() : 0) 
              << ")" << std::endl;
    

    // Add new values to 1D vector
    // push_back
    vec1D.push_back(100);
    vec1D.push_back(200);
    std::cout << "push_back() -- Size of the 1D vector: " << vec1D.size() << std::endl;
    

    // Add a new 2D vector to the 3D vector
    // push_back
    std::cout << "Size of the 3D vector: " << vec3D.size() << std::endl;
    vec3D.push_back(std::vector<std::vector<int>>());
    std::cout << "Size of the 3D vector after 2d push_back: " << vec3D.size() << std::endl;

    // Pop_back
    // pop_back is used to remove the last element from the vector
    vec1D.pop_back();
    std::cout << "pop_back() -- Size of the 1D vector: " << vec1D.size() << std::endl;

    // Remove index from vector
    // erase
    vec1D.erase(vec1D.begin() + 1);
    std::cout << "erase() -- Size of the 1D vector: " << vec1D.size() << std::endl;

    // Clear the vector
    // clear
    vec1D.clear();
    std::cout << "clear() -- Size of the 1D vector: " << vec1D.size() << std::endl;
    
    vec1D = {1, 2, 3, 4, 5};

    //Print values of Vectors
    std::cout << "1D Vector values: ";
    for (int i = 0; i < vec1D.size(); i++) {
        std::cout << vec1D[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "2D Vector values: " << std::endl;
    for (int i = 0; i < vec2D.size(); i++) {
        for (int j = 0; j < vec2D[i].size(); j++) {
            std::cout << vec2D[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "3D Vector values: " << std::endl;
    for (int i = 0; i < vec3D.size(); i++) {
        for (int j = 0; j < vec3D[i].size(); j++) {
            for (int k = 0; k < vec3D[i][j].size(); k++) {
                std::cout << vec3D[i][j][k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }


    // ***** Vector Data Access *****
    std::cout << "\n ***** Vector Access ***** \n" << std::endl;

    // 1D Vector Iteration
    std::cout << "1D Vector: ";
    for (int i : vec1D) {
        std::cout << i << " ";
    }

    // 2D Vector Iteration
    std::cout << "\n2D Vector: " << std::endl;
    for (std::vector<int> vec : vec2D) {
        for (int i : vec) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }

    // 3D Vector Iteration
    std::cout << "\n3D Vector: " << std::endl;
    for (std::vector<std::vector<int>> vec2D : vec3D) {
        for (std::vector<int> vec : vec2D) {
            for (int i : vec) {
                std::cout << i << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    
}