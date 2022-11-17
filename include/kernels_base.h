#ifndef __KERNELS_BASE_H__
#define __KERNELS_BASE_H__
#include <iostream>

using namespace cv;

/* 
A helper function to check if given app_name is supported as a kernel or not.
Since template is used, this definition needed to be located in this header 
file for successful compilation.
*/
template <typename T>
bool if_kernel_in_table(std::unordered_map<std::string, T> func_table, const std::string app_name){
    return (func_table.find(app_name) == func_table.end())?false:true;
}

template <typename T>
void kernel_existence_checking(std::unordered_map<std::string, T> func_table, const std::string app_name){
    if(!if_kernel_in_table(func_table, app_name)){
        std::cout << "app_name: " << app_name << " not found" << std::endl;
        std::cout << "supported app: " << std::endl;
        for(auto const &pair: func_table){
            std::cout << "{" << pair.first << ": " << pair.second << "}" << std::endl;
        }
        exit(0);
    }
}

/* Base class for kernels*/
class KernelBase{
public:
    /*
        The base input_conversion() API returns converted input data type from void* type.
    */
    virtual void input_conversion(){};
    
    /*
        The base output_conversion() API returns converted output data type to void* type.
    */
    virtual void output_conversion(){};
    
    /*
        The base run_kernel() API returns kernel time in millisecond in double type.
    */
    virtual double run_kernel(){
        std::cout << "virtual kernel run is not implemented yet." << std::endl;
        exit(0);
    };
};

#endif
