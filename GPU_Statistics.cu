/*
 * File:        GPU_Statistics.cu
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
 * Comments:    Implementation file of the GA statistics
 *              This class maintains and collects GA statistics
 *
 *
 * License:     This source code is distribute under OpenSouce GNU GPL license
 *
 *              If using this code, please consider citation of related papers
 *              at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 *
 *
 * Created on 24 March 2012, 00:00
 * Modified on 22 September 2021, 16:50
 */

#include <helper_cuda.h>
#include <sstream>

#include "GPU_Statistics.h"
#include "CUDA_Kernels.h"


//----------------------------------------------------------------------------//
//                              Definitions                                   //
//----------------------------------------------------------------------------//


//----------------------------------------------------------------------------//
//                       TGPU_Statistics Implementation                       //
//                              public methods                                //
//----------------------------------------------------------------------------//


/*
 * Constructor of the class
 *
 */
TGPU_Statistics::TGPU_Statistics(){

    AllocateCudaMemory();

}// end of TGPU_Population
//------------------------------------------------------------------------------


/*
 * Destructor of the class
 *
 */
TGPU_Statistics::~TGPU_Statistics(){

    FreeCudaMemory();

}// end of ~TGPU_Population
//------------------------------------------------------------------------------


/*
 /*
 * Calculate Statistics
 *
 * @param       Population - Pointer to population to calculate statistics over
 * @param       PrintBest  - If true, copy the best individual
 *
 */
void TGPU_Statistics::Calculate(TGPU_Population * Population, bool PrintBest){


    // Init statistics struct on GPU
    InitStatistics();

    //  Run a kernel to collect statistic data
    CalculateStatistics
            <<<TParameters::GetInstance()->GetGPU_SM_Count()*2, BLOCK_SIZE >>>
            (DeviceData, Population->DeviceData);
    CheckAndReportCudaError(__FILE__,__LINE__);

    // Copy statistics down to host
    CopyOut(Population, PrintBest);

    // Calculate derived statistics
    HostData->AvgFitness = HostData->AvgFitness / TParameters::GetInstance()->PopulationSize();
    HostData->Divergence = sqrtf(fabs((
             HostData->Divergence / TParameters::GetInstance()->PopulationSize()) - (HostData->AvgFitness* HostData->AvgFitness)));


}// end of Calculate
//------------------------------------------------------------------------------

/*
 * Print best individual as a string
 *
 * @param Global knapsack data
 * @retur Best individual in from of a sting
 */
string TGPU_Statistics::GetBestIndividualStr(TKnapsackData * GlobalKnapsackData){

    TParameters * Params = TParameters::GetInstance();

    /// Lambda function to convert 1 int into a bit string
    auto convertIntToBitString= [] (TGene value, int nValidDigits) -> std::string
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

    const int nBlocks = GlobalKnapsackData->OriginalNumberOfItems / 32;

    for (int blockId = 0; blockId < nBlocks; blockId++)
    {
      bestChromozome += convertIntToBitString(HostBestIndividual[blockId], 32) + "\n";
    }

    // Reminder
    if (GlobalKnapsackData->OriginalNumberOfItems % 32 > 0 )
    {
      bestChromozome += convertIntToBitString(HostBestIndividual[nBlocks],
                                              GlobalKnapsackData->OriginalNumberOfItems % 32);
    }

 return bestChromozome;
}// end of GetBestIndividualStr
//------------------------------------------------------------------------------



//----------------------------------------------------------------------------//
//                       TGPU_Statistics Implementation                       //
//                              protected methods                             //
//----------------------------------------------------------------------------//

/*
 * Allocate GPU memory
 */
void TGPU_Statistics::AllocateCudaMemory(){

    //------------ Host data ---------------//

    // Allocate basic Host structure
    checkCudaErrors(
            cudaHostAlloc((void**)&HostData,  sizeof(TStatisticsData)
                           ,cudaHostAllocDefault )
            );

    // Allocate best individual on the host side
    checkCudaErrors(
            cudaHostAlloc((void**)&HostBestIndividual,  sizeof(TGene) * TParameters::GetInstance()->ChromosomeSize()
                           ,cudaHostAllocDefault)
            );


    //------------ Device data ---------------//

    // Allocate data structure on host side
    checkCudaErrors(
           cudaMalloc((void**)&DeviceData,  sizeof(TStatisticsData))
           );



}// end of AllocateMemory
//------------------------------------------------------------------------------

/*
 * Free GPU memory
 */
void TGPU_Statistics::FreeCudaMemory(){


    // Free CPU Best individual
    checkCudaErrors(
            cudaFreeHost(HostBestIndividual)
            );

    // Free structure in host memory
    checkCudaErrors(
            cudaFreeHost(HostData)
            );


    // Free whole structure
    checkCudaErrors(
           cudaFree(DeviceData)
           );

}// end of FreeMemory
//------------------------------------------------------------------------------





/*
 * Copy data from GPU Statistics structure to CPU
 *
 * @param Population - Population where to find the best individual
 * @param PrintBest  - In false, there's no need to transfer an individual down to host
 *
 */
void TGPU_Statistics::CopyOut(TGPU_Population * Population, bool PrintBest){

    TParameters * Params = TParameters::GetInstance();

    // Copy 4 statistics values
    checkCudaErrors(
         cudaMemcpy(HostData, DeviceData, sizeof(TStatisticsData),
                    cudaMemcpyDeviceToHost)
         );

    // Copy of chromosome
    if (PrintBest){

        Population->CopyOutIndividual(HostBestIndividual, HostData->IndexBest);
    }


}// end of CopyOut
//------------------------------------------------------------------------------



/*
 * Initialize statistics before computation
 *
 */
void TGPU_Statistics::InitStatistics(){


    HostData->MaxFitness  = TFitness(0);
    HostData->MinFitness  = TFitness(UINT_MAX);
    HostData->AvgFitness  = 0.0f;
    HostData->Divergence  = 0.0f;
    HostData->IndexBest   = 0;


    // Copy 4 statistics values
    checkCudaErrors(
         cudaMemcpy(DeviceData, HostData, sizeof(TStatisticsData),
                    cudaMemcpyHostToDevice)
         );


}// end of InitStatistics
//------------------------------------------------------------------------------



