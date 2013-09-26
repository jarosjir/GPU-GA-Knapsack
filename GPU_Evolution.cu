/* 
 * File:        GPU_Evolution.cu
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
 * Comments:    Implementation file of the GA evolution
 *              This class controls the evolution process on multicore CPU
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

#include <iostream>
#include <stdio.h>
#include <sys/time.h>

#include "GPU_Evolution.h"
#include "GPU_Statistics.h"
#include "CUDA_Kernels.h"
#include "Parameters.h"

using namespace std;


//----------------------------------------------------------------------------//
//                              Definitions                                   //
//----------------------------------------------------------------------------//


//----------------------------------------------------------------------------//
//                              Implementation                                //
//                              public methods                                //
//----------------------------------------------------------------------------//

/*
 * Constructor of the class
 */
TGPU_Evolution::TGPU_Evolution(){
    
    // Create parammeter class
    Params    = TParameters::GetInstance();
    pDeviceIdx = 0;
    
    // Select device 
    cudaSetDevice(pDeviceIdx);
    
    // Get parameters of the device
    cudaDeviceProp 	prop;	
    cudaGetDeviceProperties (&prop, pDeviceIdx);
   
    pMultiprocessorCount = prop.multiProcessorCount;
    Params->SetGPU_SM_Count(prop.multiProcessorCount);
    
    
    // Load data from file
    GlobalData.LoadFromFile();
    
    // Create populations on GPU
    MasterPopulation    = new TGPU_Population(Params->PopulationSize(), Params->ChromosomeSize());    
    OffspringPopulation = new TGPU_Population(Params->OffspringPopulationSize(), Params->ChromosomeSize());
    
    MigrationPopulation_In  = new TGPU_Population(Params->EmigrantCount(), Params->ChromosomeSize());
    MigrationPopulation_Out = new TGPU_Population(Params->EmigrantCount(), Params->ChromosomeSize());
        
    // Create statistics
    GPUStatistics = new TGPU_Statistics();

    
    pActGeneration = 0;    
    
    // Initialize Random seed
    InitSeed();
    
}// end of TGPU_Evolution
//------------------------------------------------------------------------------
  

/*
 * Destructor of the class
 */
TGPU_Evolution::~TGPU_Evolution(){
    
    delete MasterPopulation;    
    delete OffspringPopulation;
    
    delete MigrationPopulation_In;
    delete MigrationPopulation_Out;
        
    delete GPUStatistics;

    
}// end of Destructor
//------------------------------------------------------------------------------

/*
 * Run Evolution
 */
void TGPU_Evolution::Run(){
    
    Initialize();
            
    RunEvolutionCycle();        
    
}// end of Run
//------------------------------------------------------------------------------


//----------------------------------------------------------------------------//
//                              Implementation                                //
//                              protected methods                             //
//----------------------------------------------------------------------------//


/*
 * Initialize seed
 */
void TGPU_Evolution::InitSeed() {

  struct timeval tp1;  
  
  gettimeofday(&tp1, NULL);
  
  pSeed = (tp1.tv_sec / (pDeviceIdx+1)) * tp1.tv_usec;
  
};// end of InitSeed
//------------------------------------------------------------------------------
    



/*
 * Initialization of the GA
 */
void TGPU_Evolution::Initialize(){
    
        
    pActGeneration = 0;
        
    // Store parameters on GPU and print them out
    Params->StoreParamsOnGPU();
    Params->PrintAllParameters();
    
    // Set elements count
       
    int Elements = Params->ChromosomeSize() * Params->PopulationSize();
                            
    //-- Initialize population --//
    FirstPopulationGenerationKernel
            <<<pMultiprocessorCount * 2, BLOCK_SIZE>>>
            (MasterPopulation->DeviceData, GetSeed());

    dim3 Blocks; 
    dim3 Threads;
            
    Blocks.x = 1;    
    Blocks.y = (Params->PopulationSize() / (CHR_PER_BLOCK) +1);
    Blocks.z = 1;
    
    
    Threads.x = WARP_SIZE;
    Threads.y = CHR_PER_BLOCK;
    Threads.z = 1;
    
        
    
    // Calculate Knapsack fintess
    CalculateKnapsackFintess
            <<<Blocks, Threads>>>
                (MasterPopulation->DeviceData, GlobalData.DeviceData);
    
     
}// end of TGPU_Evolution
//------------------------------------------------------------------------------




/*
 * Run evolutionary cycle for defined number of generations
 * 
 */
void TGPU_Evolution::RunEvolutionCycle(){
    
    
    dim3 Blocks;
    dim3 Threads;
    
    Threads.x = WARP_SIZE;
    Threads.y = CHR_PER_BLOCK;
    Threads.z = 1;
    
    
    
    // Evaluate generations
    for (pActGeneration = 1; pActGeneration < Params->NumOfGenerations(); pActGeneration++) {
      
      
          //-------------Selection -----------//
          Blocks.x = 1;    
          Blocks.y = (Params->OffspringPopulationSize() % (CHR_PER_BLOCK << 1)  == 0) ?
                            Params->OffspringPopulationSize() / (CHR_PER_BLOCK << 1)  :
                            Params->OffspringPopulationSize() / (CHR_PER_BLOCK << 1) + 1;

          Blocks.z = 1;

          GeneticManipulationKernel
                  <<<Blocks, Threads>>>
                  (MasterPopulation->DeviceData, OffspringPopulation->DeviceData, GetSeed());



          //----------- Evaluation ---------//

          Blocks.x = 1;    
          Blocks.y = (Params->OffspringPopulationSize() % (CHR_PER_BLOCK)  == 0) ?
                            Params->OffspringPopulationSize() / (CHR_PER_BLOCK)  :
                            Params->OffspringPopulationSize() / (CHR_PER_BLOCK) + 1;
          Blocks.z = 1;


          CalculateKnapsackFintess
                <<<Blocks, Threads>>>
                    (OffspringPopulation->DeviceData, GlobalData.DeviceData);




          //----------- Replacement ---------//


          Blocks.x = 1;    
          Blocks.y = (Params->PopulationSize() % (CHR_PER_BLOCK)  == 0) ?
                            Params->PopulationSize() / (CHR_PER_BLOCK)  :
                            Params->PopulationSize() / (CHR_PER_BLOCK) + 1;
          Blocks.z = 1;



          ReplacementKernel
                  <<<Blocks, Threads>>>
                  (MasterPopulation->DeviceData, OffspringPopulation->DeviceData, GetSeed());

         
          if (pActGeneration % Params->StatisticsInterval() == 0){
              GPUStatistics->Calculate(MasterPopulation, Params->GetPrintBest());
             
              printf("Generation %6d, MaxFitness %6f, MinFitness %6f, AvgFitness %6f, Diver %6f \n", 
                      pActGeneration, GPUStatistics->HostData->MaxFitness, GPUStatistics->HostData->MinFitness,
                                      GPUStatistics->HostData->AvgFitness, GPUStatistics->HostData->Divergence);
              
              if (Params->GetPrintBest())  printf("%s\n", GPUStatistics->GetBestIndividualStr(GlobalData.HostData).c_str());
          }
                  
              
                  
    }
        
    
    
                  
        GPUStatistics->Calculate(MasterPopulation, true);
        printf("------------------------------------------------------------------------------\n");
        printf("FinalMaxFitness %6f, FinalMinFitness %6f, FinalAvgFitness %6f, FinalDiver %6f \n", 
                GPUStatistics->HostData->MaxFitness, GPUStatistics->HostData->MinFitness,
                GPUStatistics->HostData->AvgFitness, GPUStatistics->HostData->Divergence);
        printf("%s\n", GPUStatistics->GetBestIndividualStr(GlobalData.HostData).c_str());
          
    
}// end of RunEvolutionCycle
//------------------------------------------------------------------------------