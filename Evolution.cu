/**
 * @file        Evolution.cu
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
 * @brief:      Implementation file of the GA evolution
 *              This class controls the evolution process on single GPU.
 *
 * @date        30 March     2012, 00:00 (created)
 *              23 September 2021, 15:51 (revised)
 *
 * @copyright   Copyright (C) 2012 - 2021 Jiri Jaros.
 *
 * This source code is distribute under OpenSouce GNU GPL license.
 * If using this code, please consider citation of related papers
 * at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 */

#include <stdio.h>
#include <sys/time.h>

#include "Evolution.h"
#include "Statistics.h"
#include "CUDAKernels.h"
#include "Parameters.h"

using namespace std;

//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------- Definitions ----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


/**
 * Constructor of the class
 */
GPUEvolution::GPUEvolution()
    : mParams(Parameters::GetInstance()),
      mActGeneration(0),
      mMultiprocessorCount(0),
      mDeviceIdx(0),
      mRandomSeed(0)
{
  // Select device
  cudaSetDevice(mDeviceIdx);
  checkAndReportCudaError(__FILE__,__LINE__);

  // Get parameters of the device
  cudaDeviceProp 	prop;
  cudaGetDeviceProperties (&prop, mDeviceIdx);
  checkAndReportCudaError(__FILE__,__LINE__);

  mMultiprocessorCount = prop.multiProcessorCount;
  mParams.setNumberOfDeviceSMs(prop.multiProcessorCount);

  // Load knapsack data from the file.
  mGlobalData.LoadFromFile();

  // Create populations on GPU
  mMasterPopulation    = new GPUPopulation(mParams.getPopulationSize(),          mParams.getChromosomeSize());
  mOffspringPopulation = new GPUPopulation(mParams.getOffspringPopulationSize(), mParams.getChromosomeSize());

  // Create statistics
  mStatistics = new GPUStatistics();

    // Initialize Random seed
  initRandomSeed();
}// end of GPUEvolution
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor of the class.
 */
GPUEvolution::~GPUEvolution()
{
  delete mMasterPopulation;
  delete mOffspringPopulation;

  delete mStatistics;
}// end of Destructor
//----------------------------------------------------------------------------------------------------------------------

/**
 * Run Evolution
 */
void GPUEvolution::run()
{
  initialize();

  runEvolutionCycle();
}// end of run
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//----------------------------------------------- Protected methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Initialize seed
 */
void GPUEvolution::initRandomSeed()
{
  struct timeval tp1;

  gettimeofday(&tp1, nullptr);

  mRandomSeed = (tp1.tv_sec / (mDeviceIdx + 1)) * tp1.tv_usec;
};// end of initRandomSeed
//----------------------------------------------------------------------------------------------------------------------

/**
 * Initialization of the GA
 */
void GPUEvolution::initialize()
{
  mActGeneration = 0;

  // Store parameters on GPU and print them out
  mParams.copyToDevice();
  mParams.printOutAllParameters();

  // Initialize population
  cudaGenerateFirstPopulationKernel<<<mMultiprocessorCount * 2, BLOCK_SIZE>>>
                                   (mMasterPopulation->getDeviceData(),
                                    getRandomSeed());
  checkAndReportCudaError(__FILE__,__LINE__);

  // Set number of blocks and threads
  dim3 blocks;
  dim3 threads;

  // Generate necessary number of blocks, for simple addressing, use Y dimension for blocks.
  blocks.x = 1;
  blocks.y = (mParams.getPopulationSize() / (CHR_PER_BLOCK) + 1);
  blocks.z = 1;

  // Every chromosome is treated by a single warp, there are as many warps as individuals per block
  threads.x = WARP_SIZE;
  threads.y = CHR_PER_BLOCK;
  threads.z = 1;

  // Calculate Knapsack fintess
  cudaCalculateKnapsackFintess<<<blocks, threads>>>
                              (mMasterPopulation->getDeviceData(), mGlobalData.getDeviceData());
  checkAndReportCudaError(__FILE__,__LINE__);

}// end of initialize
//----------------------------------------------------------------------------------------------------------------------

/**
 * Run evolutionary cycle for defined number of generations
 *
 */
void GPUEvolution::runEvolutionCycle()
{
  dim3 blocks;
  dim3 threads;

  // Every chromosome is treated by a single warp, there are as many warps as individuals per block
  threads.x = WARP_SIZE;
  threads.y = CHR_PER_BLOCK;
  threads.z = 1;

  // Evaluate generations
  for (mActGeneration = 1; mActGeneration < mParams.getNumOfGenerations(); mActGeneration++)
  {
    //--------------------------------------Selection, Crossover and Mutation ----------------------------------------//
    // Set number of blocks
    blocks.x = 1;
    blocks.y = (mParams.getOffspringPopulationSize() % (CHR_PER_BLOCK << 1)  == 0)
                  ? mParams.getOffspringPopulationSize() / (CHR_PER_BLOCK << 1)  :
                    mParams.getOffspringPopulationSize() / (CHR_PER_BLOCK << 1) + 1;
    blocks.z = 1;

    cudaGeneticManipulationKernel<<<blocks, threads>>>
                                 (mMasterPopulation->getDeviceData(),
                                  mOffspringPopulation->getDeviceData(),
                                  getRandomSeed());
    checkAndReportCudaError(__FILE__,__LINE__);

    //------------------------------------------------- Evaluation ---------------------------------------------------//
    // Set number of blocks
    blocks.x = 1;
    blocks.y = (mParams.getOffspringPopulationSize() % (CHR_PER_BLOCK)  == 0)
                  ? mParams.getOffspringPopulationSize() / (CHR_PER_BLOCK)  :
                    mParams.getOffspringPopulationSize() / (CHR_PER_BLOCK) + 1;
    blocks.z = 1;


    cudaCalculateKnapsackFintess<<<blocks, threads>>>
                                (mOffspringPopulation->getDeviceData(),
                                 mGlobalData.getDeviceData());
    checkAndReportCudaError(__FILE__,__LINE__);

    //------------------------------------------------ Replacement ---------------------------------------------------//
    blocks.x = 1;
    blocks.y = (mParams.getPopulationSize() % (CHR_PER_BLOCK)  == 0)
                  ? mParams.getPopulationSize() / (CHR_PER_BLOCK)  :
                    mParams.getPopulationSize() / (CHR_PER_BLOCK) + 1;
    blocks.z = 1;



    cudaReplacementKernel<<<blocks, threads>>>
                         (mMasterPopulation->getDeviceData(),
                          mOffspringPopulation->getDeviceData(),
                          getRandomSeed());
    checkAndReportCudaError(__FILE__,__LINE__);

    //------------------------------------------------- Statistics ---------------------------------------------------//

    if (mActGeneration % mParams.getStatisticsInterval() == 0)
    {
      mStatistics->calculate(mMasterPopulation, mParams.getPrintBest());

      printf("Generation %6d, MaxFitness %6f, MinFitness %6f, AvgFitness %6f, Diver %6f \n",
             mActGeneration,
             mStatistics->getHostData()->maxFitness,
             mStatistics->getHostData()->minFitness,
             mStatistics->getHostData()->avgFitness,
             mStatistics->getHostData()->divergence);

      if (mParams.getPrintBest())
      {
        printf("%s\n", mStatistics->getBestIndividualStr(mGlobalData.getHostData()).c_str());
      }
    }// stats
  }// evolution loop


  //----------------------------------------------- Final statistics -------------------------------------------------//
  mStatistics->calculate(mMasterPopulation, true);
  printf("------------------------------------------------------------------------------\n");
  printf("FinalMaxFitness %6f, FinalMinFitness %6f, FinalAvgFitness %6f, FinalDiver %6f \n",
         mStatistics->getHostData()->maxFitness, mStatistics->getHostData()->minFitness,
         mStatistics->getHostData()->avgFitness, mStatistics->getHostData()->divergence);
  printf("%s\n", mStatistics->getBestIndividualStr(mGlobalData.getHostData()).c_str());
}// end of runEvolutionCycle
//----------------------------------------------------------------------------------------------------------------------