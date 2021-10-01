/**
 * @file        CUDAKernels.cu
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
 * @brief       Implementation file of the GA evolution CUDA kernel
 *              This class controls the evolution process on a single GPU
 *
 * @date        30 March     2012, 00:00 (created)
 *              23 September 2021, 16:36 (revised)
 *
 * @copyright   Copyright (C) 2012 - 2021 Jiri Jaros.
 *
 * This source code is distribute under OpenSouce GNU GPL license.
 * If using this code, please consider citation of related papers
 * at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 */


#include <limits.h>
#include "Random123/philox.h"

#include "Population.h"
#include "Parameters.h"
#include "GlobalKnapsackData.h"

#include "CUDAKernels.h"

//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------- Definitions ----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/// Extern variable for constant memory.
extern __constant__  EvolutionParameters gpuEvolutionParameters;

using namespace r123;

/**
 * @class Semaphore
 * Semaphore class for reduction kernels.
 */
class Semaphore
{
  public:
    /// Default constructor.
    Semaphore() = default;
    /// Default destructor.
    ~Semaphore() = default;

    /// Acquire semaphore.
    __device__ void acquire()
    {
      while (atomicCAS((int *)&mutex, 0, 1) != 0);
      __threadfence();
    }

    /// Release semaphore.
    __device__ void release()
    {
      mutex = 0;
      __threadfence();
    }

  private:
    /// Mutex for the semaphore.
    volatile int mutex = 0;
};// end of Semaphore
//----------------------------------------------------------------------------------------------------------------------


/// Global semaphore variable.
__device__ Semaphore sempahore;


/// Datatype for two 32b random values.
typedef r123::Philox2x32 RNG_2x32;
/// Datatype for four 32b random values.
typedef r123::Philox4x32 RNG_4x32;


/**
 * Generate two random values
 * @param [in] key     - Key for the generator.
 * @param [in] counter - Counter for the generator.
 * @return two random values
 */
__device__ RNG_2x32::ctr_type generateTwoRndValues(unsigned int key,
                                                   unsigned int counter);

//
/**
 * Get index into the array
 * @params [in] chromosomeIdx - Chromosome index.
 * @params [in] geneIdx       - Gene index.
 * @returns Index into the population array.
 */
inline __device__ int getIndex(unsigned int chromosomeIdx,
                               unsigned int geneIdx);

/**
 * Select an individual from a population using Tournament selection.
 * @param [in] ParentsData  - Parent population to select from.
 * @param [in] Random1      - First random value.
 * @param [in] Random2      - Second random value.
 * @return Index of the selected individual.
 */
inline __device__ int selection(const PopulationData* parentsData,
                                unsigned int          random1,
                                unsigned int          random2);

/**
 * Perform uniform crossover on 32 genes.
 * @param [out] offspring1  - First offspring.
 * @param [out] offspring2  - Second offspring.
 * @param [in]  parent1     - First parent.
 * @param [in]  parent2     - Second parent.
 * @param [in]  mask        - Mask to perform crossover.
 */
inline __device__ void uniformCrossover(Gene&        offspring1,
                                        Gene&        offspring2,
                                        const Gene&  parent1,
                                        const Gene&  parent2,
                                        unsigned int mask);
/**
 * Perform bit flip mutation on a selected genes.
 * @param [in, out] offspring1 - first offspring to be mutated.
 * @param [in, out] offspring2 - second offspring to be mutated.
 * @param [in]      random1    - first random value.
 * @param [in]      random2    - second random values.
 * @param [in]      bitIdx     - bit to be flipped.
 */
inline __device__ void bitFlipMutation(Gene&        offspring1,
                                       Gene&        offspring2,
                                       unsigned int random1,
                                       unsigned int random2,
                                       int          bitIdx);


/**
 * Half warp reduce for price.
 *
 * @param [in, out] data      - Data to reduce.
 * @param [in]      threadIdx - idx of thread.
 */
template<class T>
__device__ void halfWarpReduce(volatile T* data,
                               int threadIdx);


//--------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------- Implementation --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Check and report CUDA errors.
 * if there's an error the code exits.
 */
void checkAndReportCudaError(const char* sourceFileName,
                             const int   sourceLineNumber)
{
  const cudaError_t cudaError = cudaGetLastError();

  if (cudaError != cudaSuccess)
  {
    fprintf(stderr,
            "Error in the CUDA routine: \"%s\"\nFile name: %s\nLine number: %d\n",
            cudaGetErrorString(cudaError),
            sourceFileName,
            sourceLineNumber);

    exit(EXIT_FAILURE);
  }
}// end of checkAndReportCudaError
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Device Functions --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Device random number generation.
 *
 */
inline __device__ RNG_2x32::ctr_type generateTwoRndValues(unsigned int key,
                                                          unsigned int counter)
{
    RNG_2x32 rng;

    return rng({0, counter}, {key});
}// end of TwoRandomINTs
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get index into population.
 */
inline __device__ int getIndex(unsigned int chromosomeIdx,
                               unsigned int geneIdx)
{
  return (chromosomeIdx * gpuEvolutionParameters.chromosomeSize + geneIdx);
}// end of getIndex
//----------------------------------------------------------------------------------------------------------------------

/**
 * Half warp reduce.
 */
template<class T>
__device__ void halfWarpReduce(volatile T* data, int threadIdx)
{
  if (threadIdx < WARP_SIZE / 2)
  {
    data[threadIdx] += data[threadIdx + 16];
    //__syncwarp();
    data[threadIdx] += data[threadIdx + 8];
    //__syncwarp();
    data[threadIdx] += data[threadIdx + 4];
    //__syncwarp();
    data[threadIdx] += data[threadIdx + 2];
    //__syncwarp();
    data[threadIdx] += data[threadIdx + 1];
  }
}// end of halfWarpReduce
//----------------------------------------------------------------------------------------------------------------------

/**
 * Select one individual.
 */
inline __device__ int selection(const PopulationData* parentsData,
                                unsigned int          random1,
                                unsigned int          random2)
{
  unsigned int idx1 = random1 % (parentsData->populationSize);
  unsigned int idx2 = random2 % (parentsData->populationSize);

  return (parentsData->fitness[idx1] > parentsData->fitness[idx2]) ? idx1 : idx2;
}// selection
//----------------------------------------------------------------------------------------------------------------------

/**
 * Uniform Crossover.
 * Flip bites of parents to produce parents
 */
inline __device__ void uniformCrossover(Gene&        offspring1,
                                        Gene&        offspring2,
                                        const Gene&  parent1,
                                        const Gene&  parent2,
                                        unsigned int mask)
{
  offspring1 = (~mask & parent1) | ( mask  & parent2);
  offspring2 = ( mask & parent1) | (~mask  & parent2);
}// end of uniformCrossover
//----------------------------------------------------------------------------------------------------------------------

/**
 * BitFlip Mutation.
 */
inline __device__ void bitFlipMutation(Gene&        offspring1,
                                       Gene&        offspring2,
                                       unsigned int random1,
                                       unsigned int random2,
                                       int          bitIdx)
{
  if (random1 < gpuEvolutionParameters.mutationUintBoundary)
  {
    offspring1 ^= (1 << bitIdx);
  }
  if (random2 < gpuEvolutionParameters.mutationUintBoundary)
  {
    offspring2 ^= (1 << bitIdx);
  }
}// end of bitFlipMutation
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------- CUDA kernels ----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Initialize Population before run.
 */
__global__ void cudaGenerateFirstPopulationKernel(PopulationData* populationData,
                                                  unsigned int    randomSeed)
{
   size_t i      = threadIdx.x + blockIdx.x * blockDim.x;
   size_t stride = blockDim.x * gridDim.x;

   const int nGenes = populationData->chromosomeSize * populationData->populationSize;

   // Generate random genes.
   while (i < nGenes)
   {
      const RNG_2x32::ctr_type randomValues = generateTwoRndValues(i, randomSeed);
      populationData->population[i] = randomValues.v[0];

      i += stride;
      if (i < nGenes)
      {
        populationData->population[i] = randomValues.v[1];
      }
      i += stride;
    }

   // Zero fitness values
   i  = threadIdx.x + blockIdx.x * blockDim.x;
   while (i < populationData->populationSize)
   {
      populationData->fitness[i] = 0.0f;
      i += stride;
   }
}// end of cudaGenerateFirstPopulationKernel
//----------------------------------------------------------------------------------------------------------------------

/**
 * Genetic Manipulation (Selection, Crossover, Mutation)
 *
 */
__global__ void cudaGeneticManipulationKernel(const PopulationData* parentsData,
                                              PopulationData*       offspringData,
                                              unsigned int          randomSeed)
{
  const int geneIdx = threadIdx.x;
  const int chromosomeIdx = 2 * (threadIdx.y + blockIdx.y * blockDim.y);

  // Init random generator.
  RNG_4x32  rng_4x32;
  RNG_4x32::key_type key     = {{static_cast<unsigned int>(geneIdx), static_cast<unsigned int>(chromosomeIdx)}};
  RNG_4x32::ctr_type counter = {{0, 0, randomSeed ,0xbeeff00d}};
  RNG_4x32::ctr_type randomValues;

  // If having enough offsprings, return
  if (chromosomeIdx >= gpuEvolutionParameters.offspringPopulationSize)
  {
    return;
  }

  // Produce new offspring

  __shared__ int  parent1Idx[CHR_PER_BLOCK];
  __shared__ int  parent2Idx[CHR_PER_BLOCK];
  __shared__ bool crossoverFlag[CHR_PER_BLOCK];


  //---------------------------------------------- selection ---------------------------------------------------------//

  if ((threadIdx.y == 0) && (threadIdx.x < CHR_PER_BLOCK))
  {
    counter.incr();
    randomValues = rng_4x32(counter, key);

    parent1Idx[threadIdx.x] = selection(parentsData, randomValues.v[0], randomValues.v[1]);
    parent2Idx[threadIdx.x] = selection(parentsData, randomValues.v[2], randomValues.v[3]);

    counter.incr();
    randomValues = rng_4x32(counter, key);
    crossoverFlag[threadIdx.x] = randomValues.v[0] < gpuEvolutionParameters.crossoverUintBoundary;
  }

  __syncthreads();

  //-------------------------------------------- Manipulation  -------------------------------------------------------//

  // Go through two chromosomes and do uniform crossover and mutation
  for (int geneIdx = threadIdx.x; geneIdx < gpuEvolutionParameters.chromosomeSize; geneIdx += WARP_SIZE)
  {
    const Gene geneParent1 = parentsData->population[getIndex(parent1Idx[threadIdx.y], geneIdx)];
    const Gene geneParent2 = parentsData->population[getIndex(parent2Idx[threadIdx.y], geneIdx)];

    Gene geneOffspring1 = 0;
    Gene geneOffspring2 = 0;

    // Crossover
    if (crossoverFlag[threadIdx.y])
    {
      counter.incr();
      randomValues = rng_4x32(counter, key);
      uniformCrossover(geneOffspring1, geneOffspring2, geneParent1, geneParent2, randomValues.v[0]);
    }
    else
    {
      geneOffspring1 = geneParent1;
      geneOffspring2 = geneParent2;
    }

    // Mutation --//
    for (int bitID = 0; bitID < gpuEvolutionParameters.intBlockSize; bitID += 2)
    {
      counter.incr();
      randomValues = rng_4x32(counter, key);

      bitFlipMutation(geneOffspring1, geneOffspring2, randomValues.v[0], randomValues.v[1], bitID);
      bitFlipMutation(geneOffspring1, geneOffspring2, randomValues.v[2], randomValues.v[3], bitID + 1);
    }// for

    offspringData->population[getIndex(chromosomeIdx    , geneIdx)] = geneOffspring1;
    offspringData->population[getIndex(chromosomeIdx + 1, geneIdx)] = geneOffspring2;
  }
}// end of cudaGeneticManipulationKernel
//----------------------------------------------------------------------------------------------------------------------


/**
 * Replacement kernel.
 */
__global__ void cudaReplacementKernel(const PopulationData * parentsData,
                                      PopulationData*        offspringData,
                                      unsigned int           randomSeed)
{
  const int chromosomeIdx = threadIdx.y + blockIdx.y * blockDim.y;

  // Init random generator.
  RNG_2x32::ctr_type randomValues;
  __shared__ unsigned int offspringIdx[CHR_PER_BLOCK];

  // If having enogugh offsprings, return.
  if (chromosomeIdx >= gpuEvolutionParameters.populationSize)
  {
    return;
  }

  // Select offspring
  if (threadIdx.x == 0)
  {
    randomValues = generateTwoRndValues(chromosomeIdx, randomSeed);
    offspringIdx[threadIdx.y] = randomValues.v[0] % (gpuEvolutionParameters.offspringPopulationSize);
  }

  __syncthreads();


  // Replacement
  if (parentsData->fitness[chromosomeIdx] < offspringData->fitness[offspringIdx[threadIdx.y]])
  {
    //-- copy data --//
    for (int geneIdx = threadIdx.x; geneIdx < gpuEvolutionParameters.chromosomeSize; geneIdx += WARP_SIZE)
    {
      parentsData->population[getIndex(chromosomeIdx, geneIdx)]
              = offspringData->population[getIndex(offspringIdx[threadIdx.y], geneIdx)];

    }

    if (threadIdx.x == 0)
    {
      parentsData->fitness[chromosomeIdx] = offspringData->fitness[offspringIdx[threadIdx.y]];
    }
  } // Replacement
}// end of cudaReplacementKernel
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate statistics
 */
__global__ void cudaCalculateStatistics(StatisticsData*       statisticsData,
                                        const PopulationData* populationData)
{
  __shared__ Fitness sharedMax[BLOCK_SIZE];
  __shared__ int     sharedMaxIdx[BLOCK_SIZE];
  __shared__ Fitness sharedMin[BLOCK_SIZE];

  __shared__ float sharedSum[BLOCK_SIZE];
  __shared__ float sharedSum2[BLOCK_SIZE];


  //Clear shared buffers
  sharedMax[threadIdx.x]    = Fitness(0);
  sharedMaxIdx[threadIdx.x] = 0;
  sharedMin[threadIdx.x]    = Fitness(UINT_MAX);

  sharedSum[threadIdx.x]  = 0.0f;;
  sharedSum2[threadIdx.x] = 0.0f;;

  __syncthreads();

  Fitness fitnessValue;

  // Reduction to shared memory
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < gpuEvolutionParameters.populationSize;
       i += blockDim.x * gridDim.x)
  {
    fitnessValue = populationData->fitness[i];
    if (fitnessValue > sharedMax[threadIdx.x])
    {
      sharedMax[threadIdx.x]    = fitnessValue;
      sharedMaxIdx[threadIdx.x] = i;
    }

    if (fitnessValue < sharedMin[threadIdx.x])
    {
      sharedMin[threadIdx.x] = fitnessValue;
    }

    sharedMin[threadIdx.x] = min(sharedMin[threadIdx.x], fitnessValue);

    sharedSum[threadIdx.x]  += fitnessValue;
    sharedSum2[threadIdx.x] += fitnessValue * fitnessValue;
  }

  __syncthreads();

  // Reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
	  if (threadIdx.x < stride)
    {
      if (sharedMax[threadIdx.x] < sharedMax[threadIdx.x + stride])
      {
        sharedMax[threadIdx.x]    = sharedMax[threadIdx.x + stride];
        sharedMaxIdx[threadIdx.x] = sharedMaxIdx[threadIdx.x + stride];
      }
      if (sharedMin[threadIdx.x] > sharedMin[threadIdx.x + stride])
      {
        sharedMin[threadIdx.x] = sharedMin[threadIdx.x + stride];
      }
      sharedSum[threadIdx.x]  += sharedSum[threadIdx.x + stride];
      sharedSum2[threadIdx.x] += sharedSum2[threadIdx.x + stride];
    }
	__syncthreads();
  }

  __syncthreads();


  // Write to Global Memory using a single thread per block and a semaphore
  if (threadIdx.x == 0)
  {
    sempahore.acquire();

    if (statisticsData->maxFitness < sharedMax[threadIdx.x])
    {
      statisticsData->maxFitness = sharedMax[threadIdx.x];
      statisticsData->indexBest  = sharedMaxIdx[threadIdx.x];
    }

    if (statisticsData->minFitness > sharedMin[threadIdx.x])
    {
      statisticsData->minFitness = sharedMin[threadIdx.x];
    }

    sempahore.release();

    atomicAdd(&(statisticsData->avgFitness), sharedSum [threadIdx.x]);
    atomicAdd(&(statisticsData->divergence), sharedSum2[threadIdx.x]);
  }
}// end of cudaCalculateStatistics
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate Knapsack fitness.
 *
 */
__global__ void cudaCalculateKnapsackFintess(PopulationData*     populationData,
                                             const KnapsackData* globalData)
{
  __shared__ PriceType  priceGlobalData[WARP_SIZE];
  __shared__ WeightType weightGlobalData[WARP_SIZE];

  __shared__ PriceType  priceValues[CHR_PER_BLOCK][WARP_SIZE];
  __shared__ WeightType weightValues[CHR_PER_BLOCK][WARP_SIZE];


  const int geneInBlockIdx = threadIdx.x;
  const int chromosomeIdx  = threadIdx.y + blockIdx.y * blockDim.y;

  // If not having anything to evaluate, return.
  if (chromosomeIdx >= populationData->populationSize)
  {
    return;
  }

  priceValues[threadIdx.y][threadIdx.x]  = PriceType(0);
  weightValues[threadIdx.y][threadIdx.x] = WeightType(0);

  //-- calculate weight and price in parallel
  for (int intBlockIdx = 0; intBlockIdx < gpuEvolutionParameters.chromosomeSize; intBlockIdx++)
  {
    // Load Data
    if (threadIdx.y == 0)
    {
      priceGlobalData[geneInBlockIdx]  = globalData->itemPrice [intBlockIdx * gpuEvolutionParameters.intBlockSize
                                                                + geneInBlockIdx];
      weightGlobalData[geneInBlockIdx] = globalData->itemWeight[intBlockIdx * gpuEvolutionParameters.intBlockSize
                                                                + geneInBlockIdx];
    }

    const Gene actGene = ((populationData->population[getIndex(chromosomeIdx, intBlockIdx)]) >> geneInBlockIdx) &
                          Gene(1);

    __syncthreads();

    // Calculate Price and Weight

    priceValues[threadIdx.y][geneInBlockIdx]  += actGene * priceGlobalData[geneInBlockIdx];
    weightValues[threadIdx.y][geneInBlockIdx] += actGene * weightGlobalData[geneInBlockIdx];
  }

  // Everything above is warp synchronous.
  __syncwarp();
  halfWarpReduce(priceValues [threadIdx.y], threadIdx.x);
  halfWarpReduce(weightValues[threadIdx.y], threadIdx.x);
  __syncwarp();

  // write the result
  if (threadIdx.x == 0)
  {
    Fitness result = Fitness(priceValues[threadIdx.y][0]);

    // Penalize
    if (weightValues[threadIdx.y][0] > globalData->knapsackCapacity)
    {
      const Fitness penalty = (weightValues[threadIdx.y][0] - globalData->knapsackCapacity);

      result = result - globalData->maxPriceWightRatio * penalty;
      if (result < 0 ) result = Fitness(0);
    }

    populationData->fitness[chromosomeIdx] = result;
   } // if
}// end of cudaCalculateKnapsackFintess
//----------------------------------------------------------------------------------------------------------------------

