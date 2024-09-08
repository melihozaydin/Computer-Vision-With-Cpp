## https://stackoverflow.com/questions/74231905/how-to-convert-eigen-martix-to-torch-tensor

I'm new to C++. I use Libtorch and Eigen and want to convert Eigen::Martix to Torch::Tensor, but I could not.

I wrote the code below refering to https://github.com/andrewssobral/dtt
```
#include <torch/torch.h>
#include <Eigen/Dense>
#include <iostream>

int main(){
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> E(1,3);
    E(0,0) = 1;
    E(0,1) = 2;
    E(0,2) = 3;
    std::cout << "E" << std::endl;
    std::cout << E << std::endl;

    std::vector<int64_t> dims = {E.rows(), E.cols()};
    torch::Tensor T = torch::from_blob(E.data(), dims).clone();
    std::cout << "T" << std::endl;
    std::cout << T << std::endl;
}
```
I run this code and get the output below.
```
E
1 2 3
T
 0.0000  1.8750  0.0000
[ CPUFloatType{1,3} ]
```
But I expect output to be:
```
E
1 2 3
T
1 2 3
[ CPUFloatType{1,3} ]
```
--------------


By default, libtorch will imagine that the tensor you are creating is a float tensor (see at the end of the print : CPUFloatType). Since your eigen matrix is double, you want to change this behavior like this :
```
auto options = torch::TensorOptions().dtype(torch::kDouble);
torch::Tensor T = torch::from_blob(E.data(), dims, options).clone();
```
