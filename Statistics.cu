/**
 * @file        Statistics.cu
 * @author      Jiri Jaros
 *              Brno University of Technology
 *              Faculty of Information Technology
 *
 *              and
 *
 *              The Australian National University
 *              ANU College of Engineering & Computer Science
 *
 *              jarosjir@fit.vutbr.cz
 *              www.fit.vutbr.cz/~jarosjir
 *
 * @brief       Implementation file of the GA statistics
 *              This class maintains and collects GA statistics
 *
 * @date        30 March     2012, 00:00 (created)
 *              22 September 2021, 19:59 (revised)
 *
 * @copyright   Copyright (C) 2012 - 2021 Jiri Jaros.
 *
 * This source code is distribute under OpenSouce GNU GPL license.
 * If using this code, please consider citation of related papers
 * at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 */

#include <helper_cuda.h>
#include <sstream>

#include "Statistics.h"
#include "CUDAKernels.h"


//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------- Definitions ----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


/**
 * Constructor of the class
 *
 */
GPUStatistics::GPUStatistics()
{
  allocateMemory();
}// end of GPUStatistics
//----------------------------------------------------------------------------------------------------------------------

/*
 * Destructor of the class
 *
 */
GPUStatistics::~GPUStatistics()
{
  freeMemory();
}// end of ~GPUStatistics
//----------------------------------------------------------------------------------------------------------------------


/**
 * Calculate Statistics.
 */
void GPUStatistics::calculate(GPUPopulation* population,
                              bool             printBest)
{
  const Parameters& params = Parameters::GetInstance();

  // Init statistics struct on GPU
  initStatistics();

  //  Run a kernel to collect statistic data.
  cudaCalculateStatistics<<<params.getNumberOfDeviceSMs() * 2, BLOCK_SIZE >>>
                         (deviceData, population->getDeviceData());
  checkAndReportCudaError(__FILE__,__LINE__);

  // Copy statistics down to host
  copyFromDevice(population, printBest);

  // Calculate derived statistics
  hostData->avgFitness = hostData->avgFitness / params.getPopulationSize();

  hostData->divergence = sqrt(fabs((hostData->divergence / params.getPopulationSize()) -
                                   (hostData->avgFitness * hostData->avgFitness)));

}// end of calculate
//----------------------------------------------------------------------------------------------------------------------

/**
 * Print best individual as a string.
 */
std::string GPUStatistics::getBestIndividualStr(const KnapsackData* GlobalKnapsackData) const
{
  /// Lambda function to convert 1 int into a bit string
  auto convertIntToBitString= [] (Gene value, int nValidDigits) -> std::string
  {
    std::string str = "";

    for (int bit = 0; bit < nValidDigits; bit++)
    {
      str += ((value & (1 << bit)) == 0) ? "0" : "1";
      str += (bit % 8 == 7) ? " " : "";
    }

    for (int bit = nValidDigits; bit < 32; bit++)
    {
      str += 'x';
      str += (bit % 8 == 7) ? " " : "";
    }

    return str;
  };// end of convertIntToBitString

  std::string bestChromozome = "";

  const int nBlocks = GlobalKnapsackData->originalNumberOfItems / 32;

  for (int blockId = 0; blockId < nBlocks; blockId++)
  {
    bestChromozome += convertIntToBitString(hostBestIndividual[blockId], 32) + "\n";
  }

  // Reminder
  if (GlobalKnapsackData->originalNumberOfItems % 32 > 0 )
  {
    bestChromozome += convertIntToBitString(hostBestIndividual[nBlocks],
                                            GlobalKnapsackData->originalNumberOfItems % 32);
  }

 return bestChromozome;
}// end of getBestIndividualStr
//------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------------------------//
//----------------------------------------------- Protected methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Allocate memory.
 */
void GPUStatistics::allocateMemory()
{
  //------------ Host data ---------------//

  // Allocate basic Host structure
  checkCudaErrors(cudaHostAlloc<StatisticsData>(&hostData,  sizeof(StatisticsData), cudaHostAllocDefault));

  // Allocate best individual on the host side
  checkCudaErrors(cudaHostAlloc<Gene>(&hostBestIndividual,
                                      sizeof(Gene) * Parameters::GetInstance().getChromosomeSize(),
                                      cudaHostAllocDefault));


  //------------ Device data ---------------//

  // Allocate data structure on host side
  checkCudaErrors(cudaMalloc<StatisticsData>(&deviceData,  sizeof(StatisticsData)));
}// end of allocateMemory
//----------------------------------------------------------------------------------------------------------------------

/*
 * Free GPU memory.
 */
void GPUStatistics::freeMemory()
{
  // Free CPU Best individual
  checkCudaErrors(cudaFreeHost(hostBestIndividual));

  // Free structure in host memory
  checkCudaErrors(cudaFreeHost(hostData));

  // Free whole structure
  checkCudaErrors(cudaFree(deviceData));
}// end of freeMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy data from GPU Statistics structure to CPU.
 */
void GPUStatistics::copyFromDevice(GPUPopulation* population,
                                   bool             printBest)
{
  const Parameters& params = Parameters::GetInstance();

  // Copy 4 statistics values
  checkCudaErrors(cudaMemcpy(hostData, deviceData, sizeof(StatisticsData), cudaMemcpyDeviceToHost));

  // Copy of chromosome
  if (printBest)
  {
    population->copyIndividualFromDevice(hostBestIndividual, hostData->indexBest);
  }
}// end of copyFromDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Initialize statistics before computation.
 */
void GPUStatistics::initStatistics()
{
  hostData->maxFitness  = Fitness(0);
  hostData->minFitness  = Fitness(UINT_MAX);
  hostData->avgFitness  = 0.0f;
  hostData->divergence  = 0.0f;
  hostData->indexBest   = 0;

  // Copy 4 statistics values
  checkCudaErrors(cudaMemcpy(deviceData, hostData, sizeof(StatisticsData), cudaMemcpyHostToDevice));
}// end of initStatistics
//----------------------------------------------------------------------------------------------------------------------

