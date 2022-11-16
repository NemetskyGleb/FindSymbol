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

constexpr int TAB_SIZE = 256;

__global__ void countSymbolsKernel(const char* data, uint32_t size, int* countsTab)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        atomicAdd(&countsTab[data[i]], 1);
    }
}

void PrintResult(const int* countsTab)
{
    std::cout << "Symbols count:\n";
    int32_t totalCount = 0;
    for (int i = 0; i < TAB_SIZE; i++)
    {
        if (countsTab[i])
        {
            std::cout << (char)i << " : " << countsTab[i] << std::endl;
            totalCount += countsTab[i];
        }
    }
    std::cout << "Total count: " << totalCount << '\n';
}

void countSymbolsCpu(const char* data, uint32_t size,  int* countsTab,
    std::function<void(const char* data, uint32_t size, int* countsTab)> countSymbols)
{
    using namespace std::chrono;

    auto start = high_resolution_clock::now();
    
    countSymbols(data, size, countsTab);

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "Time taken by function: "
        << duration.count() << " milliseconds" << std::endl;

    PrintResult(countsTab);
}

void countSymbols(const char* data, uint32_t size, int* countsTab)
{
    for (size_t i = 0; i < size; i++)
    {
        countsTab[data[i]]++;
    }
}

void countSymbolsCuda(const char* data, uint32_t length, int* countsTab);

void CompareResults(const int* tab1, const int* tab2);

int main()
{
    std::string filePath;
    
    std::cout << "Enter name of file: ";
    std::cin >> filePath;

    std::ifstream input_file(filePath, std::ifstream::in);
    if (!input_file.is_open())
    {
        std::cerr << "Can't open file.\n";
        std::exit(-1);
    }

    std::stringstream buf;

    buf << input_file.rdbuf();

    std::cout << "Result on CPU:" << std::endl;

    uint32_t length = static_cast<uint32_t>( buf.str().size());

    int countsTabCpu[TAB_SIZE] = { 0 };
    int* countsTabGpu = new int[TAB_SIZE];

    countSymbolsCpu(buf.str().data(), length, countsTabCpu, &countSymbols);
    
    // filling with nulls
    std::fill_n(countsTabGpu, TAB_SIZE, 0);

    countSymbolsCuda(buf.str().data(), length, countsTabGpu);

    PrintResult(countsTabGpu);

    CompareResults(countsTabCpu, countsTabGpu);

    delete countsTabGpu;

    return 0;
}


void countSymbolsCuda(const char* data, uint32_t length, int* countsTab)
{
    char* dev_data;
    int* dev_tab;

    cudaError_t cudaStatus;

    auto checkError = [&](cudaError_t status)
    {
        if (status != cudaSuccess)
        {
            std::cerr << "Error! ";
            std::cerr << cudaGetErrorString(status) << std::endl;
            cudaFree(dev_data);
            cudaFree(dev_tab);
            if (countsTab) {
                delete[] countsTab;
            }
            exit(-1);
        }
    };

    cudaStatus = cudaSetDevice(0);
    checkError(cudaStatus);
    
    cudaStatus = cudaMalloc((void**)&dev_data, length );
    checkError(cudaStatus);
    cudaStatus = cudaMalloc((void**)&dev_tab, TAB_SIZE * sizeof(int));
    checkError(cudaStatus);

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cudaStatus = cudaMemcpy(dev_data, data, length, cudaMemcpyHostToDevice);
    checkError(cudaStatus);

    countSymbolsKernel<<<(int)ceil((float)length / T), T>>>(dev_data, length, dev_tab);

    cudaStatus = cudaGetLastError();
    checkError(cudaStatus);

    cudaStatus = cudaDeviceSynchronize();
    checkError(cudaStatus);

    cudaStatus = cudaMemcpy(countsTab, dev_tab, TAB_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    checkError(cudaStatus);

    cudaStatus = cudaEventRecord(stop, 0);
    checkError(cudaStatus);

    cudaStatus = cudaEventSynchronize(stop);
    checkError(cudaStatus);
    cudaStatus = cudaEventElapsedTime(&elapsedTime, start, stop);
    checkError(cudaStatus);

    cudaStatus = cudaEventDestroy(start);
    checkError(cudaStatus);
    cudaStatus = cudaEventDestroy(stop);
    checkError(cudaStatus);

    printf("Time spent executing by the GPU: %.2f milliseconds\n", elapsedTime);

    cudaStatus = cudaDeviceReset();
    checkError(cudaStatus);

    cudaFree(dev_data);
    cudaFree(dev_tab);
}

void CompareResults(const int* tab1, const int* tab2)
{
    bool equal = true;
    for (int i = 0; i < TAB_SIZE; i++)
    {
        if (tab1[i] != tab2[i])
        {
            equal = false;
        }
    }
    if (equal)
    {
        std::cout << "Results are equal!" << std::endl;
    }
    else {
        std::cout << "Result aren't equal!" << std::endl;
    }
}