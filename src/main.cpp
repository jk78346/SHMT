#include <iostream>
#include <quality.h>
#include <gptpu.h>

int main(){
    std::cout << "main" << std::endl;

    std::string path = "./xxx.tflite";

    int* in;
    int* out;

    run_a_model(path, 1, in, 1, out, 1, 1);

    return 0;
}
