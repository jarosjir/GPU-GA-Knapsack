/* 
 * File:        CUDA_Kernels.h
 * Author:      Jiri Jaros
 * Affiliation: Brno University of Technology
 *              Faculty of Information Technology
 *              
 *              and
 * 
 *              The Australian National University
 *              ANU College of Engineering & Computer Science
 *
 * Email:       jarosjir@fit.vutbr.cz
 * Web:         www.fit.vutbr.cz/~jarosjir
 * 
 * Comments:    Header file of the GA evolution CUDA kernel
 *              This class controls the evolution process on a single GPU
 *
 * 
 * License:     This source code is distribute under OpenSouce GNU GPL license
 *                
 *              If using this code, please consider citation of related papers
 *              at http://www.fit.vutbr.cz/~jarosjir/pubs.php        
 *      
 *
 * 
 * Created on 30 March 2012, 00:00 PM
 */


#include <limits.h>
#include "Random123/philox.h"
#include <stdio.h>
#include <cutil_inline.h>

#include "GPU_Population.h"
#include "Parameters.h"
#include "GlobalKnapsackData.h"


#include "CUDA_Kernels.h"

//----------------------------------------------------------------------------//
//                              Definitions                                   //
//----------------------------------------------------------------------------//

__constant__  TEvolutionParameters GPU_EvolutionParameters;



using namespace r123;


typedef r123::Philox2x32 RNG_2x32;
typedef r123::Philox4x32 RNG_4x32;


// Generate two random numbers
__device__ void TwoRandomINTs (RNG_2x32::ctr_type *RandomValues, unsigned int Key, unsigned int Counter);


// Get index of array
inline __device__ int  GetIndex(unsigned int ChromosomeIdx, unsigned int GeneIdx);

// Select an individual from a population
inline __device__ int  Selection(TPopulationData * ParentsData, unsigned int Random1, unsigned int Random2);

// Perform Uniform Crossover
inline __device__ void CrossoverUniformFlip(TGene& GeneOffspring1, TGene& GeneOffspring2,
                                            TGene GeneParent1    , TGene GeneParent2,
                                            unsigned int RandomValue);

// Perform BitFlip mutation
inline __device__ void MutationBitFlip(TGene& GeneOffspring1, TGene& GeneOffspring2,
                                       unsigned int RandomValue1,unsigned int RandomValue2, int BitID);


// Reduction kernels
__device__ void HalfWarpReducePrice (volatile TPriceType * sdata, int tid);
__device__ void HalfWarpReduceWeight(volatile TWeightType* sdata, int tid);
__device__ void HalfWarpReduceGene  (volatile TGene* sdata, int tid);


//----------------------------------------------------------------------------//
//                              Kernel implementation                         //
//----------------------------------------------------------------------------//



//----------------------------------------------------------------------------//
//                              GPU Lock implementation                         //
//----------------------------------------------------------------------------//


/*
 * Constructor of the lock
 */
TGPU_Lock::TGPU_Lock(void){
    
    int state = 0;
    cutilSafeCall(cudaMalloc((void**)& mutex, sizeof(int)));
    cutilSafeCall(cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice));
    
}// end of TGPU_Lock

//------------------------------------------------------------------------------

/*
 * Destructor of the lock
 */
TGPU_Lock::~TGPU_Lock(void){
    
    cudaFree(mutex);
        
}// end of ~TGPU_Lock
//------------------------------------------------------------------------------

/*
 * Lock the lock
 * 
 */
__device__ void TGPU_Lock::Lock(void){
    
    while (atomicCAS(mutex, 0, 1) != 0 );
    
}// end of Lock
//------------------------------------------------------------------------------


/*
 * Unlock the lock
 * 
 */
__device__ void TGPU_Lock::Unlock( void ){
    
   atomicExch(mutex, 0);
       
}// end of Unlock
//------------------------------------------------------------------------------





//----------------------------------------------------------------------------//
//                          DeviceFunctions Kernels                           //
//----------------------------------------------------------------------------//



/*
 * Device random number generation
 * 
 * @param RandomValues - Returned random values
 * @param Key
 * @param Counter
 * 
 */
inline __device__ void TwoRandomINTs(RNG_2x32::ctr_type *RandomValues, 
                                     unsigned int Key, unsigned int Counter){
    RNG_2x32 rng;

    RNG_2x32::ctr_type counter={{0,Counter}};
    RNG_2x32::key_type key={{Key}};
            
    *RandomValues = rng(counter, key);           
}// end of TwoRandomINTs
//------------------------------------------------------------------------------



/*
 * GetIndex to population
 * @param ChromosomeIdx
 * @param genIdx
 * @return 1D index
 * 
 */
inline __device__ int  GetIndex(unsigned int ChromosomeIdx, unsigned int GeneIdx){
       
    return (ChromosomeIdx * GPU_EvolutionParameters.ChromosomeSize + GeneIdx);
    
}// end of GetIndex
//------------------------------------------------------------------------------


/*
 * Half warp reduce for price
 * 
 * @param sdata  - data to reduce
 * @param tid    - idx of thread
 * 
 */
__device__ void HalfWarpReducePrice(volatile TPriceType * sdata, int tid){
    if (tid < WARP_SIZE/2) {
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];    
    }
}// end of HalfWarpReducePrice
//------------------------------------------------------------------------------


/*
 * Half warp reduce for Weight
 *
 * @param sdata  - data to reduce
 * @param tid    - idx of thread
 * 
 */
__device__ void HalfWarpReduceWeight(volatile TWeightType* sdata, int tid){
    if (tid < WARP_SIZE/2) {
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];        
    }
}// end of HalfWarpReducePrice
//------------------------------------------------------------------------------


/*
 * Half Warp reduction for TGene
 *
 * @param sdata  - data to reduce
 * @param tid    - idx of thread 
 */
__device__ void HalfWarpReduceGene(volatile TGene* sdata, int tid){    
    if (tid < WARP_SIZE/2) {
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];        
    }
}// end of HalfWarpReduceGene
//------------------------------------------------------------------------------


//----------------------------------------------------------------------------//
//                                DeviceFunctions Kernels                     //
//----------------------------------------------------------------------------//



/*
 * Select one individual
 * @param ParentData - Parent population
 * @param Random1    - First random value
 * @param Random2    - Second random value
 * @return           - idx of the selected individual
 */
inline __device__ int Selection(TPopulationData * ParentsData, unsigned int Random1, unsigned int Random2){
   

    unsigned int Idx1 = Random1 % (ParentsData->PopulationSize);
    unsigned int Idx2 = Random2 % (ParentsData->PopulationSize);
    
    return (ParentsData->Fitness[Idx1] > ParentsData->Fitness[Idx2]) ? Idx1 : Idx2;
}// Selection
//------------------------------------------------------------------------------


/*
 * Uniform Crossover
 * Flip bites of parents to produce parents
 * 
 * @param       GeneOffspring1 - Returns first offspring (one gene)
 * @param       GeneOffspring2 - Returns second offspring (one gene)
 * @param       GeneParent1    - First parent (one gene)
 * @param       GeneParent2    - Second parent (one gene)
 * @param       Mask           - Random value for mask
 * 
 */
inline __device__ void CrossoverUniformFlip(TGene& GeneOffspring1, TGene& GeneOffspring2,
                                            TGene GeneParent1    , TGene GeneParent2,
                                            unsigned int Mask){

    
    GeneOffspring1 =  (~Mask  & GeneParent1) | ( Mask  & GeneParent2);
    GeneOffspring2 =  ( Mask  & GeneParent1) | (~Mask  & GeneParent2);
    
    
}// end of CrossoverUniformFlip
//------------------------------------------------------------------------------



/*
 * BitFlip Mutation
 * Invert selected bit
 * 
 * @param       GeneOffspring1 - Returns first offspring (one gene)
 * @param       GeneOffspring2 - Returns second offspring (one gene)
 * @param       RandomValue1   - Random value 1
 * @param       RandomValue2   - Random value 2
 * @param       BitID          - Bit to mutate
 
 */
inline __device__ void MutationBitFlip(TGene& GeneOffspring1, TGene& GeneOffspring2,
                                      unsigned int RandomValue1,unsigned int RandomValue2, int BitID){
        
  //GeneOffspring1 ^= ((unsigned int)(RandomValue1 < GPU_EvolutionParameters.MutationUINTBoundary) << BitID); 
  //GeneOffspring2 ^= ((unsigned int)(RandomValue2 < GPU_EvolutionParameters.MutationUINTBoundary) << BitID); 
  if (RandomValue1 < GPU_EvolutionParameters.MutationUINTBoundary) GeneOffspring1 ^= (1 << BitID); 
  if (RandomValue2 < GPU_EvolutionParameters.MutationUINTBoundary) GeneOffspring2 ^= (1 << BitID);  
    
  // minimal influence caused by positive prediction
    
}// end of MutationBitFlip
//------------------------------------------------------------------------------
            

//----------------------------------------------------------------------------//
//                                Global Kernels                              //
//----------------------------------------------------------------------------//





/*
 * Initialize Population before run
 * @params   Population
 * @params   RandomNumbers
 * 
 */
__global__ void FirstPopulationGenerationKernel(TPopulationData * PopData, unsigned int RandomSeed){
    
   size_t i      = threadIdx.x + blockIdx.x*blockDim.x;
   size_t stride = blockDim.x * gridDim.x;
    
   RNG_2x32::ctr_type RandomValues;
    
   const int PopulationDIM = PopData->ChromosomeSize * PopData->PopulationSize;
   
   while (i < PopulationDIM) {
       
       TwoRandomINTs(&RandomValues, i, RandomSeed);             
       PopData->Population[i] = RandomValues.v[0];
              
       i += stride;
       if (i < PopulationDIM) {
          PopData->Population[i] = RandomValues.v[1];
       }
       i += stride;       
    }  
   
   i  = threadIdx.x + blockIdx.x*blockDim.x;
   while (i < PopData->PopulationSize){
        PopData->Fitness[i]    = 0.0f;
        i += stride;
   }
   
}// end of PopulationInitializationKernel
//------------------------------------------------------------------------------






/*
 * Genetic Manipulation (Selection, Crossover, Mutation)
 * 
 * @param ParentsData
 * @param OffspringData
 * @param RandomSeed
 * 
 */
__global__ void GeneticManipulationKernel(TPopulationData * ParentsData, TPopulationData * OffspringData, 
                                    unsigned int RandomSeed){
    int GeneIdx       = threadIdx.x;
    int ChromosomeIdx = 2* (threadIdx.y + blockIdx.y * blockDim.y); 
    
    //-- Init Random --//
    RNG_4x32  rng_4x32;    
    RNG_4x32::key_type key    ={{GeneIdx, ChromosomeIdx}};
    RNG_4x32::ctr_type counter={{0, 0, RandomSeed ,0xbeeff00d}};
    RNG_4x32::ctr_type RandomValues;
    
    
    
    
    if (ChromosomeIdx >= GPU_EvolutionParameters.OffspringPopulationSize) return;
    
    __shared__ int  Parent1_Idx  [CHR_PER_BLOCK];
    __shared__ int  Parent2_Idx  [CHR_PER_BLOCK];
    __shared__ bool CrossoverFlag[CHR_PER_BLOCK];
    
    //------------------------------------------------------------------------//
    //------------------------ selection -------------------------------------//
    //------------------------------------------------------------------------//
    if ((threadIdx.y == 0) && (threadIdx.x < CHR_PER_BLOCK)){        
        counter.incr();
        RandomValues = rng_4x32(counter, key);
        
        Parent1_Idx[threadIdx.x] = Selection(ParentsData, RandomValues.v[0], RandomValues.v[1]);                        
        Parent2_Idx[threadIdx.x] = Selection(ParentsData, RandomValues.v[2], RandomValues.v[3]);
                
        counter.incr();
        RandomValues = rng_4x32(counter, key);
        CrossoverFlag[threadIdx.x] = RandomValues.v[0] < GPU_EvolutionParameters.CrossoverUINTBoundary;        
    }
    

    __syncthreads();
    //------------------------------------------------------------------------//
    //------------------------ Manipulation  ---------------------------------//
    //------------------------------------------------------------------------//

    //-- Go through two chromosomes and do uniform crossover and mutation--//
    while (GeneIdx < GPU_EvolutionParameters.ChromosomeSize){        
        TGene GeneParent1 = ParentsData->Population[GetIndex(Parent1_Idx[threadIdx.y], GeneIdx)];
        TGene GeneParent2 = ParentsData->Population[GetIndex(Parent2_Idx[threadIdx.y], GeneIdx)];

        TGene GeneOffspring1 = 0;
        TGene GeneOffspring2 = 0;

        //-- crossover --//
        if (CrossoverFlag[threadIdx.y]) {

            counter.incr();
            RandomValues = rng_4x32(counter, key);
            CrossoverUniformFlip(GeneOffspring1, GeneOffspring2, GeneParent1, GeneParent2, RandomValues.v[0]);            
            
        } else {
            GeneOffspring1 = GeneParent1;
            GeneOffspring2 = GeneParent2;
        }
            

        //-- mutation --//
        for (int BitID = 0; BitID < GPU_EvolutionParameters.IntBlockSize; BitID+=2){                

            counter.incr();            
            RandomValues = rng_4x32(counter, key);

            MutationBitFlip(GeneOffspring1, GeneOffspring2, RandomValues.v[0],RandomValues.v[1], BitID);
            MutationBitFlip(GeneOffspring1, GeneOffspring2, RandomValues.v[2],RandomValues.v[3], BitID+1);


         }// for

        OffspringData->Population[GetIndex(ChromosomeIdx  , GeneIdx)] = GeneOffspring1;
        OffspringData->Population[GetIndex(ChromosomeIdx+1, GeneIdx)] = GeneOffspring2;

        GeneIdx += WARP_SIZE;
    }
           
           
}// end of GeneticManipulation
//------------------------------------------------------------------------------




/*
 * Replacement kernel (Selection, Crossover, Mutation)
 * 
 * @param ParentsData
 * @param OffspringData
 * @param RandomSeed
 */
__global__ void ReplacementKernel(TPopulationData * ParentsData, TPopulationData * OffspringData, unsigned int RandomSeed){
   
    
    
    int GeneIdx       = threadIdx.x;
    int ChromosomeIdx = threadIdx.y + blockIdx.y * blockDim.y; 
        
    //-- Init Random --//
    RNG_2x32::ctr_type RandomValues;      
    __shared__ unsigned int OffspringIdx_SHM[CHR_PER_BLOCK];
        
    if (ChromosomeIdx >= GPU_EvolutionParameters.PopulationSize) return;

    
    //-- select offspring --//
    if (threadIdx.x == 0){
       TwoRandomINTs(&RandomValues, ChromosomeIdx, RandomSeed);                     
       OffspringIdx_SHM[threadIdx.y]  = RandomValues.v[0] % (GPU_EvolutionParameters.OffspringPopulationSize);
              
    }
    
    __syncthreads();
    
    
    //------- replacement --------//
    if (ParentsData->Fitness[ChromosomeIdx] < OffspringData->Fitness[OffspringIdx_SHM[threadIdx.y]]){
              
        //-- copy data --//
        while (GeneIdx < GPU_EvolutionParameters.ChromosomeSize){                
            ParentsData->Population[GetIndex(ChromosomeIdx, GeneIdx)] = OffspringData->Population[GetIndex(OffspringIdx_SHM[threadIdx.y], GeneIdx)];                   
            GeneIdx +=  WARP_SIZE;
        }    
    
        if (threadIdx.x == 0) ParentsData->Fitness[ChromosomeIdx] = OffspringData->Fitness[OffspringIdx_SHM[threadIdx.y]];
                        
    } // replacement      
    
}// end of ReplacementKernel
//------------------------------------------------------------------------------



/*
 * Calculate statistics
 * 
 * @param StatisticsData
 * @param PopopulationData
 * @param GPULock
 * 
 */
__global__ void CalculateStatistics(TStatisticsData * StatisticsData, TPopulationData * PopData, TGPU_Lock GPULock){
    
  int i      = threadIdx.x + blockDim.x*blockIdx.x;
  int stride = blockDim.x*gridDim.x;
  
  __shared__ TFitness shared_Max    [BLOCK_SIZE];
  __shared__ int      shared_Max_Idx[BLOCK_SIZE];
  __shared__ TFitness shared_Min    [BLOCK_SIZE];
  
  __shared__ float shared_Sum    [BLOCK_SIZE];
  __shared__ float shared_Sum2   [BLOCK_SIZE];
  
  
    //-- Null shared buffer --//
  
  shared_Max    [threadIdx.x] = TFitness(0);
  shared_Max_Idx[threadIdx.x] = 0;
  shared_Min    [threadIdx.x] = TFitness(UINT_MAX);
  
  shared_Sum    [threadIdx.x] = 0.0f;;
  shared_Sum2   [threadIdx.x] = 0.0f;;
  
  __syncthreads();
  
  TFitness FitnessValue;
  
  //-- Reduction to shared memory --//
  while (i < GPU_EvolutionParameters.PopulationSize){
      
      FitnessValue = PopData->Fitness[i];
      if (FitnessValue > shared_Max[threadIdx.x]){
          shared_Max    [threadIdx.x] = FitnessValue;
          shared_Max_Idx[threadIdx.x] = i;
      }
      
      if (FitnessValue < shared_Min[threadIdx.x]){
          shared_Min    [threadIdx.x] = FitnessValue;          
      }
      
      shared_Sum [threadIdx.x] += FitnessValue;
      shared_Sum2[threadIdx.x] += FitnessValue*FitnessValue;
      
      i += stride;
  }
  
  __syncthreads();
  
  //-- Reduction in shared memory --//
     
  for (int stride = blockDim.x/2; stride > 0; stride /= 2){
	if (threadIdx.x < stride) {
            if (shared_Max[threadIdx.x] < shared_Max[threadIdx.x + stride]){
               shared_Max    [threadIdx.x] = shared_Max    [threadIdx.x + stride];
               shared_Max_Idx[threadIdx.x] = shared_Max_Idx[threadIdx.x + stride];
            }
            if (shared_Min[threadIdx.x] > shared_Min[threadIdx.x + stride]){
               shared_Min [threadIdx.x] = shared_Min[threadIdx.x + stride];               
            }                
            shared_Sum [threadIdx.x] += shared_Sum [threadIdx.x + stride];
            shared_Sum2[threadIdx.x] += shared_Sum2[threadIdx.x + stride];                        
        }
	__syncthreads();
  }
  
  __syncthreads();

  
  
  //-- Write to Global Memory --//  
  if (threadIdx.x == 0){
      GPULock.Lock();
      
      if (StatisticsData->MaxFitness < shared_Max[threadIdx.x]){
               StatisticsData->MaxFitness = shared_Max    [threadIdx.x];
               StatisticsData->IndexBest  = shared_Max_Idx[threadIdx.x];                
      }
      
      if (StatisticsData->MinFitness > shared_Min[threadIdx.x]){
               StatisticsData->MinFitness = shared_Min[threadIdx.x];               
      }                
      
      StatisticsData->AvgFitness += shared_Sum [threadIdx.x];
      StatisticsData->Divergence += shared_Sum2[threadIdx.x];
              
      GPULock.Unlock();        
              
  }
    
}// end of CalculateStatistics
//------------------------------------------------------------------------------









/*
 * Calculate Knapsack fitness
 * 
 * Each warp working with 1 32b gene. Diferent warps different individuals
 * 
 * @param PopData
 * @param GlobalData
 * 
 */
__global__ void CalculateKnapsackFintess(TPopulationData * PopData, TKnapsackData * GlobalData){
    
       
    __shared__ TPriceType  PriceGlobalData_SHM [WARP_SIZE];
    __shared__ TWeightType WeightGlobalData_SHM[WARP_SIZE];

    
    __shared__ TPriceType  PriceValues_SHM [CHR_PER_BLOCK] [WARP_SIZE];
    __shared__ TWeightType WeightValues_SHM[CHR_PER_BLOCK] [WARP_SIZE];
    
    
    int GeneInBlockIdx = threadIdx.x;
    int ChromosomeIdx  = threadIdx.y + blockIdx.y * blockDim.y; 

    if (ChromosomeIdx >= PopData->PopulationSize) return;
    
    TGene ActGene;
    
    //------------------------------------------------------//
    
    PriceValues_SHM [threadIdx.y] [threadIdx.x] = TPriceType(0);
    WeightValues_SHM[threadIdx.y] [threadIdx.x] = TWeightType(0);
    
    
    
    //-- Calculate weight and price in parallel
    for (int IntBlockIdx = 0; IntBlockIdx < GPU_EvolutionParameters.ChromosomeSize; IntBlockIdx++){
                
                //--------------Load Data -------------//
        if (threadIdx.y == 0) {        
                PriceGlobalData_SHM [GeneInBlockIdx] = GlobalData->ItemPrice [IntBlockIdx * GPU_EvolutionParameters.IntBlockSize + GeneInBlockIdx];
                WeightGlobalData_SHM[GeneInBlockIdx] = GlobalData->ItemWeight[IntBlockIdx * GPU_EvolutionParameters.IntBlockSize + GeneInBlockIdx];
        }
        
        ActGene = ((PopData->Population[GetIndex(ChromosomeIdx, IntBlockIdx)]) >> GeneInBlockIdx) & TGene(1);
        
        __syncthreads();
        
        //-- Calculate Price and Weight --//
        
        PriceValues_SHM [threadIdx.y] [GeneInBlockIdx] += ActGene * PriceGlobalData_SHM  [GeneInBlockIdx];
        WeightValues_SHM[threadIdx.y] [GeneInBlockIdx] += ActGene * WeightGlobalData_SHM [GeneInBlockIdx];                
    }
    
     //------------------------------------------------------//
     //--    PER WARP computing - NO BARRIRER NECSSARY     --//
     //------------------------------------------------------//
    
    //__syncthreads(); 
    HalfWarpReducePrice (PriceValues_SHM [threadIdx.y], threadIdx.x);
    HalfWarpReduceWeight(WeightValues_SHM[threadIdx.y], threadIdx.x);
    
    //__syncthreads(); 
    
    //------------------------------------------------------//
    //--    PER WARP computing - NO BARRIRER NECSSARY     --//
    //------------------------------------------------------//

    // threadIdx.x ==0 calculate final Fitness --//
    if (threadIdx.x == 0){
        
        TFitness result = TFitness(PriceValues_SHM [threadIdx.y][0]);       
        
        
        if (WeightValues_SHM[threadIdx.y][0] > GlobalData->KnapsackCapacity){
            TFitness Penalty = (WeightValues_SHM[threadIdx.y][0] - GlobalData->KnapsackCapacity);
                        
            result = result  - GlobalData->MaxPriceWightRatio * Penalty;            
            if (result < 0 ) result = TFitness(0);            
            //result = TFitness(0);
        }
        
        PopData->Fitness[ChromosomeIdx] = result;
                
           
   } // if

   
}// end of CalculateKnapsackFintess
//-----------------------------------------------------------------------------

