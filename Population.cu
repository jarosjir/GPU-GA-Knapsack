/**
 * @file        Population.h
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
 * @brief       Implementation file of the GA population
 *              This class maintains and GA populations
 *
 * @date        30 March     2012, 00:00 (created)
 *              23 September 2021, 12:26 (revised)
 *
 * @copyright   Copyright (C) 2012 - 2021 Jiri Jaros.
 *
 * This source code is distribute under OpenSouce GNU GPL license.
 * If using this code, please consider citation of related papers
 * at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 */

#include <stdio.h>
#include <stdexcept>
#include <helper_cuda.h>

#include "Population.h"
#include "CUDAKernels.h"


//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------- Definitions ----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- GPUPopulation ----------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor of the class.
 */
GPUPopulation::GPUPopulation(const int populationSize,
                             const int chromosomeSize)
{
  mHostPopulationHandler.chromosomeSize = chromosomeSize;
  mHostPopulationHandler.populationSize = populationSize;

  allocateMemory();
}// end of GPUPopulation
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor of the class
 *
 */
GPUPopulation::~GPUPopulation()
{
  freeMemory();
}// end of GPUPopulation
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy data from CPU population structure to GPU
 * Both population must have the same size (sizes not being copied)!!
 *
 * @param HostSource - Source of population data on the host side
 */
void GPUPopulation::copyToDevice(const PopulationData* hostPopulatoin)
{
  // Basic data check
  if (hostPopulatoin->chromosomeSize != mHostPopulationHandler.chromosomeSize)
  {
    throw std::out_of_range("Wrong chromosome size in GPUPopulation::copyToDevice function.");
  }

  if (hostPopulatoin->populationSize != mHostPopulationHandler.populationSize)
  {
    throw std::out_of_range("Wrong population size in GPUPopulation::copyToDevice function.");
  }

  // Copy chromosomes
  checkCudaErrors(
        cudaMemcpy(mHostPopulationHandler.population,
                   hostPopulatoin->population,
                   sizeof(Gene) * mHostPopulationHandler.chromosomeSize * mHostPopulationHandler.populationSize,
                   cudaMemcpyHostToDevice)
        );

  // Copy fitness values
  checkCudaErrors(
        cudaMemcpy(mHostPopulationHandler.fitness,
                   hostPopulatoin->fitness,
                   sizeof(Fitness) * mHostPopulationHandler.populationSize,
                   cudaMemcpyHostToDevice)
        );


}// end of copyToDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy data from GPU population structure to CPU.
 */
void GPUPopulation::copyFromDevice(PopulationData * hostPopulatoin)
{
  if (hostPopulatoin->chromosomeSize != mHostPopulationHandler.chromosomeSize)
  {
    throw std::out_of_range("Wrong chromosome size in GPUPopulation::copyFromDevice function.");
  }

    if (hostPopulatoin->populationSize != mHostPopulationHandler.populationSize)
    {
      throw std::out_of_range("Wrong population size in GPUPopulation::copyFromDevice function.");
    }

  // Copy chromosomes
  checkCudaErrors(
        cudaMemcpy(hostPopulatoin->population,
                   mHostPopulationHandler.population,
                   sizeof(Gene) * mHostPopulationHandler.chromosomeSize * mHostPopulationHandler.populationSize,
                   cudaMemcpyDeviceToHost)
        );


  // Copy fitness values
  checkCudaErrors(
         cudaMemcpy(hostPopulatoin->fitness,
                    mHostPopulationHandler.fitness,
                    sizeof(Fitness) * mHostPopulationHandler.populationSize,
                    cudaMemcpyDeviceToHost)
         );
}// end of copyFromDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy data from different population (both on the same GPU).
 */
void GPUPopulation::copyOnDevice(const GPUPopulation* sourceDevicePopulation)
{
  if (sourceDevicePopulation->mHostPopulationHandler.chromosomeSize != mHostPopulationHandler.chromosomeSize)
  {
    throw std::out_of_range("Wrong chromosome size in GPUPopulation::copyOnDevice function.");
   }

  if (sourceDevicePopulation->mHostPopulationHandler.populationSize != mHostPopulationHandler.populationSize)
  {
    throw std::out_of_range("Wrong population size in GPUPopulation::copyOnDevice function.");
  }

  // Copy chromosomes
  checkCudaErrors(
         cudaMemcpy(mHostPopulationHandler.population,
                    sourceDevicePopulation->mHostPopulationHandler.population,
                    sizeof(Gene) * mHostPopulationHandler.chromosomeSize * mHostPopulationHandler.populationSize,
                    cudaMemcpyDeviceToDevice)
         );


  // Copy fintess values
  checkCudaErrors(
         cudaMemcpy(mHostPopulationHandler.fitness,
                    sourceDevicePopulation->mHostPopulationHandler.fitness,
                    sizeof(Fitness) * mHostPopulationHandler.populationSize,
                    cudaMemcpyDeviceToDevice)
         );

}// end of copyOnDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy a given individual from device to host.
 */
void GPUPopulation::copyIndividualFromDevice(Gene* individual,
                                             int   index)
{
  checkCudaErrors(
         cudaMemcpy(individual,
                    &(mHostPopulationHandler.population[index * mHostPopulationHandler.chromosomeSize]),
                    sizeof(Gene) * mHostPopulationHandler.chromosomeSize,
                    cudaMemcpyDeviceToHost)
         );

}// end of copyIndividualFromDevice
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- GPUPopulation ----------------------------------------------------//
//----------------------------------------------- Protected methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Allocate GPU memory.
 */
void GPUPopulation::allocateMemory()
{
  // Allocate data structure
  checkCudaErrors(cudaMalloc<PopulationData>(&mDeviceData,  sizeof(PopulationData)));


  // Allocate Population data
  checkCudaErrors(cudaMalloc<Gene>(&(mHostPopulationHandler.population),
                                     sizeof(Gene) * mHostPopulationHandler.chromosomeSize *
                                        mHostPopulationHandler.populationSize)
           );

  // Allocate Fitness data
  checkCudaErrors(
           cudaMalloc<Fitness>(&(mHostPopulationHandler.fitness),
                                sizeof(Fitness) * mHostPopulationHandler.populationSize)
           );

    // Copy structure to GPU
  checkCudaErrors(
           cudaMemcpy(mDeviceData, &mHostPopulationHandler, sizeof(PopulationData),cudaMemcpyHostToDevice )
           );

}// end of allocateMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Free memory.
 */
void GPUPopulation::freeMemory()
{
  // Free population data
  checkCudaErrors(cudaFree(mHostPopulationHandler.population));

  //Free Fitness data
  checkCudaErrors(cudaFree(mHostPopulationHandler.fitness));


  // Free whole structure
  checkCudaErrors(cudaFree(mDeviceData));

}// end of freeMemory
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- GPUPopulation ----------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor of the class.
 */
CPUPopulation::CPUPopulation(const int populationSize,
                             const int chromosomeSize)
{
  mHostData = new(PopulationData);
  mHostData->chromosomeSize = chromosomeSize;
  mHostData->populationSize = populationSize;

  allocateMemory();
}// end of CPUPopulation
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor of the class.
 */
CPUPopulation::~CPUPopulation()
{
  freeMemory();

  delete mHostData;
}// end of ~CPUPopulation
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- GPUPopulation ----------------------------------------------------//
//----------------------------------------------- Protected methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Allocate memory.
 */
void CPUPopulation::allocateMemory()
{
  // Allocate Population on the host side
  checkCudaErrors(
          cudaHostAlloc<Gene>(&mHostData->population,
                              sizeof(Gene) * mHostData->chromosomeSize * mHostData->populationSize,
                              cudaHostAllocDefault )
          );

    // Allocate fitness on the host side
    checkCudaErrors(
            cudaHostAlloc<Fitness>(&mHostData->fitness,
                                   sizeof(Fitness) *  mHostData->populationSize,
                                   cudaHostAllocDefault)
            );
}// end of allocateMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Free memory.
 */
void CPUPopulation::freeMemory()
{
  // Free population on the host side
  checkCudaErrors(cudaFreeHost(mHostData->population));

  // Free fitness on the host side
  checkCudaErrors(cudaFreeHost(mHostData->fitness));
}// end of freeMemory
//----------------------------------------------------------------------------------------------------------------------