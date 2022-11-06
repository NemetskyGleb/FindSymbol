#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include <functional>
#include <algorithm>
#include <unordered_map>
#include <fstream>

constexpr int T = 1024; // max threads per block

void substractWithCuda(int* c, const int* a, const int* b, uint32_t size);

__global__ void CountSymbolsKernel(const char* data, size_t size, std::unordered_map<char, int32_t>& result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    auto it = result.find(data[i]);
    if (it != result.end())
    {
        atomicAdd(&result[data[i]], 1);
    }
    else
    {
        result[data[i]] = 1;
    }
}

void PrintResult(const std::unordered_map<char, int32_t>& result)
{
    std::cout << "Symbols count:\n";
    int32_t totalCount = 0;
    for (const auto& elem : result)
    {
        std::cout << elem.first << " : " << elem.second << std::endl;
        totalCount += elem.second;
    }
    std::cout << "Total count: " << totalCount << '\n';
}

void calculateFunctionTime(const char* data, size_t size, 
    std::function<std::unordered_map<char, int32_t>(const char* data, size_t size)> countSymbols)
{
    using namespace std::chrono;

    auto start = high_resolution_clock::now();
    
    auto counts = countSymbols(data, size);

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "Time taken by function: "
        << duration.count() << " milliseconds" << std::endl;

    PrintResult(counts);
}

std::unordered_map<char, int32_t> CountSymbols(const char* data, size_t size)
{
    std::unordered_map<char, int32_t> result;

    for (size_t i = 0; i < size; i++)
    {
        auto it = result.find(data[i]);
        if (it != result.end())
        {
            result[data[i]]++;
        }
        else
        {
            result[data[i]] = 1;
        }
    }
    return result;
}



int main()
{
    std::ifstream input_file("data.txt", std::ifstream::in);
    if (!input_file.is_open())
    {
        std::cerr << "Can't open file.\n";
        std::exit(-1);
    }

    std::stringstream buf;

    buf << input_file.rdbuf();

    std::cout << "Result on CPU:" << std::endl;

    calculateFunctionTime(buf.str().data(), buf.str().size(), &CountSymbols);

    //GenerateText();

    //CountSymbols();
    //PrintResult(a, b, c1, arraySize);

    //delete[] c1;
    //substractWithCuda(c2, a, b, arraySize);


    //PrintResult(a, b, c2, arraySize);

    //cudaError_t cudaStatus = cudaDeviceReset();

    //delete[] c2, a, b;
    

    return 0;
}

//Helper function for using CUDA to substract vectors in parallel.
void substractWithCuda(int* c, const int* a, const int* b, uint32_t size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    auto checkError = [&](cudaError_t status)
    {
        if (status != cudaSuccess)
        {
            std::cerr << "Error! ";
            std::cerr << cudaGetErrorString(status) << std::endl;
            cudaFree(dev_c);
            cudaFree(dev_a);
            cudaFree(dev_b);
            std::exit(-1);
        }
    };

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    checkError(cudaStatus);

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    checkError(cudaStatus);
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    checkError(cudaStatus);
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    checkError(cudaStatus);

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    checkError(cudaStatus);
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    checkError(cudaStatus);

    // инициализируем события
    cudaEvent_t start, stop;
    float elapsedTime;
    // создаем события
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // запись события
    cudaEventRecord(start, 0);

    // Launch a kernel on the GPU with one thread for each element.
    substractKernel <<<(int)ceil((float)size / T), T >>>(dev_c, dev_a, dev_b, size);

    cudaStatus = cudaEventRecord(stop, 0);
    checkError(cudaStatus);
    // ожидание завершения работы ядра
    cudaStatus = cudaEventSynchronize(stop);
    checkError(cudaStatus);
    cudaStatus = cudaEventElapsedTime(&elapsedTime, start, stop);
    checkError(cudaStatus);
    // вывод информации
    printf("Time spent executing by the GPU: %.2f millseconds\n", elapsedTime);
    // уничтожение события
    cudaStatus = cudaEventDestroy(start);
    checkError(cudaStatus);
    cudaStatus = cudaEventDestroy(stop);
    checkError(cudaStatus);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    checkError(cudaStatus);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    checkError(cudaStatus);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    checkError(cudaStatus);

    // Free resources.
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}
