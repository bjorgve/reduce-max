#include "reduce-max.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>

int main(int argc, char** argv) {
  if(argc != 3) {
    std::cout << "Usage: " << argv[0] << " <matrix size> <maximum number>\n";
    return 1;
  }

  int n = atoi(argv[1]);
  float knownMax = atof(argv[2]);

  float *h_idata;
  h_idata = new float[n];

  // Initialize random seed
  srand(static_cast<unsigned>(time(0)));

  // Initialize the input data with random numbers
  for (int i = 0; i < n; i++) {
    h_idata[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX / 100.0);
  }

  // Overwrite a random index with a known larger number
  int randomIndex = rand() % n;
  h_idata[randomIndex] = knownMax;

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  std::vector<float *> d_data(deviceCount);
  std::vector<float *> d_results(deviceCount);
  std::vector<float> h_results(deviceCount);

  int chunk_size = n / deviceCount;

  for (int device = 0; device < deviceCount; device++) {
    cudaSetDevice(device);
    cudaMalloc(&d_data[device], chunk_size * sizeof(float));
    cudaMalloc(&d_results[device], sizeof(float));
    cudaMemcpy(d_data[device], &h_idata[device * chunk_size], chunk_size * sizeof(float), cudaMemcpyHostToDevice);
    find_max(d_data[device], d_results[device], chunk_size);
    cudaMemcpy(&h_results[device], d_results[device], sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Max from device " << device << " = " << h_results[device] << std::endl;
  }

  float max_val = *std::max_element(h_results.begin(), h_results.end());
  std::cout << "Max from array is " << max_val << std::endl;
  std::cout << "Expected max from array is " << knownMax << std::endl;

  // Free allocated memory
  for (int device = 0; device < deviceCount; device++) {
    cudaFree(d_data[device]);
    cudaFree(d_results[device]);
  }

  delete[] h_idata;

  return 0;
}
