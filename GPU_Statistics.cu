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
 * Created on 24 March 2012, 00:00 PM
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
    
    // Create a binary MUTEX (memory lock)
    TGPU_Lock GPU_Lock;   
    
    // Init statistics struct on GPU
    InitStatistics();
    
    //  Run a kernel to collect statistic data       
    CalculateStatistics
            <<<TParameters::GetInstance()->GetGPU_SM_Count()*2, BLOCK_SIZE >>>
            (DeviceData, Population->DeviceData, GPU_Lock);
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

    stringstream  S; 
    
    TParameters * Params = TParameters::GetInstance();
    

    // Convert by eight bits
    for (int BlockID=0; BlockID < Params->ChromosomeSize() -1; BlockID++){
     
         for (int BitID = 0; BitID < Params->IntBlockSize() -1; BitID++ ) {         
             char c = ((HostBestIndividual[BlockID] & (1 << BitID)) == 0) ? '0' : '1';         
             S << c;
             if (BitID % 8 ==7) S << " ";
         }    

         S << "\n";
     
    }
 
     // Convert the remainder
    for (int BitID = 0; BitID < Params->IntBlockSize() - (GlobalKnapsackData->NumberOfItems - GlobalKnapsackData->OriginalNumberOfItems); BitID++) {
         char c =  ((HostBestIndividual[Params->ChromosomeSize() -1] & (1 << BitID)) == 0) ? '0' : '1';
         S << c;
         if (BitID % 8 ==7) S << " ";
    }
          
 
 return S.str();   
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



