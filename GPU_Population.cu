/* 
 * File:        GPU_Population.cu
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
 * Comments:    Implementation file of the GA population
 *              This class maintains and GA populations
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


#include <stdio.h>
#include <stdexcept>
#include <cutil_inline.h>
#include <sstream>

#include "GPU_Population.h"
#include "CUDA_Kernels.h"





//----------------------------------------------------------------------------//
//                              Definitions                                   //
//----------------------------------------------------------------------------//


//----------------------------------------------------------------------------//
//                       TGPU_Population Implementation                       //
//                              public methods                                //
//----------------------------------------------------------------------------//

/*
 * Constructor of the class
 *
 * @param PopulationSize
 * @param ChromosomeSize
 * 
 */
TGPU_Population::TGPU_Population(const int PopulationSize, const int ChromosomeSize){
    
    FHost_Handlers.ChromosomeSize = ChromosomeSize;
    FHost_Handlers.PopulationSize = PopulationSize;
    
    AllocateCudaMemory();
            
}// end of TGPU_Population
//------------------------------------------------------------------------------


/*
 * Destructor of the class
 * 
 */
TGPU_Population::~TGPU_Population(){
    
    FreeCudaMemory();
           
}// end of TGPU_Population
//------------------------------------------------------------------------------



/*
 * Copy data from CPU population structure to GPU 
 * Both population must have the same size (sizes not being copied)!!
 * 
 * @param HostSource - Source of population data on the host side
 */    
void TGPU_Population::CopyIn(const TPopulationData * HostSource){
    
    //-- Basic data check --//
    if (HostSource->ChromosomeSize != FHost_Handlers.ChromosomeSize) {
        throw std::out_of_range("Wrong chromosome size in CopyIn function");
    }            
                
    if (HostSource->PopulationSize != FHost_Handlers.PopulationSize){
        throw std::out_of_range("Wrong population size in CopyIn function");        
    }
    
    // Copy chromosomes 
    cutilSafeCall( 
         cudaMemcpy(FHost_Handlers.Population,
                    HostSource->Population, sizeof(TGene) * FHost_Handlers.ChromosomeSize * FHost_Handlers.PopulationSize, 
                    cudaMemcpyHostToDevice)
         );    
    
    
    // Copy fitness values
    cutilSafeCall( 
         cudaMemcpy(FHost_Handlers.Fitness, 
                    HostSource->Fitness, sizeof(TFitness) * FHost_Handlers.PopulationSize, 
                    cudaMemcpyHostToDevice)
         );    

    
}// end of CopyIn
//------------------------------------------------------------------------------
    
    
    
 /*
 * Copy data from GPU population structure to CPU
 * Both population must have the same size (sizes not copied)!!
 * 
 * @param HostDestination - Source of population data on the host side
 */   
void TGPU_Population::CopyOut (TPopulationData * HostDestination){
    
    if (HostDestination->ChromosomeSize != FHost_Handlers.ChromosomeSize) {
        throw std::out_of_range("Wrong chromosome size in CopyOut function");
    }            
                
    if (HostDestination->PopulationSize != FHost_Handlers.PopulationSize){
        throw std::out_of_range("Wrong population size in CopyOut function");        
    }
    
    // Copy chromosomes --//
    cutilSafeCall( 
         cudaMemcpy(HostDestination->Population, FHost_Handlers.Population, 
                    sizeof(TGene) * FHost_Handlers.ChromosomeSize * FHost_Handlers.PopulationSize, 
                    cudaMemcpyDeviceToHost)
         );    
    
    
    //-- Copy fitnesses --//
    cutilSafeCall( 
         cudaMemcpy(HostDestination->Fitness, 
                    FHost_Handlers.Fitness, sizeof(TFitness) * FHost_Handlers.PopulationSize, 
                    cudaMemcpyDeviceToHost)
         );    
  
    
    
}// end of CopyOut
//------------------------------------------------------------------------------


/*
 * Copy out only one individual 
 * 
 * @param Individual - mem where to store individual
 * @param Index      - the index of individual
 * 
 */
void TGPU_Population::CopyOutIndividual(TGene * Individual, int Index){
    
    cutilSafeCall( 
         cudaMemcpy(Individual, &(FHost_Handlers.Population[Index * FHost_Handlers.ChromosomeSize]), 
                    sizeof(TGene) * FHost_Handlers.ChromosomeSize, cudaMemcpyDeviceToHost)
         );    
    
}// end of CopyOutIndividual
//------------------------------------------------------------------------------




/*
 * Copy data from different population (both on the same GPU)
 * No size check!!!
 * 
 * @param GPUPopulation - the source population
 */    
void TGPU_Population::CopyDeviceIn(const TGPU_Population * GPUPopulation){    
   
        
    if (GPUPopulation->FHost_Handlers.ChromosomeSize != FHost_Handlers.ChromosomeSize) {
        throw std::out_of_range("Wrong chromosome size in CopyIn function");
    }            
                
    if (GPUPopulation->FHost_Handlers.PopulationSize != FHost_Handlers.PopulationSize){
        throw std::out_of_range("Wrong population size in CopyIn function");        
    }
    
    // Copy chromosomes 
    cutilSafeCall( 
         cudaMemcpy(FHost_Handlers.Population, GPUPopulation->FHost_Handlers.Population, sizeof(TGene) * FHost_Handlers.ChromosomeSize * FHost_Handlers.PopulationSize, 
                    cudaMemcpyDeviceToDevice)
         );    
    
    
    // Copy fintess values
    cutilSafeCall( 
         cudaMemcpy(FHost_Handlers.Fitness, GPUPopulation->FHost_Handlers.Fitness, sizeof(TFitness) * FHost_Handlers.PopulationSize, 
                    cudaMemcpyDeviceToDevice)
         );    
   
}// end of CopyDeviceIn
//------------------------------------------------------------------------------


//----------------------------------------------------------------------------//
//                       TGPU_Population Implementation                       //
//                            protected methods                               //
//----------------------------------------------------------------------------//
    
/*
 * Allocate GPU memory
 */
void TGPU_Population::AllocateCudaMemory(){
    
    
    
    // Allocate data structure
    cutilSafeCall( 
           cudaMalloc((void**)&DeviceData,  sizeof(TPopulationData))
           );		
    
    
    // Allocate Population data
    cutilSafeCall( 
           cudaMalloc((void**)&(FHost_Handlers.Population),  sizeof(TGene) * FHost_Handlers.ChromosomeSize * FHost_Handlers.PopulationSize)
           );		
    
    // Allocate Fitness data
    cutilSafeCall( 
           cudaMalloc((void**)&(FHost_Handlers.Fitness),  sizeof(TFitness) * FHost_Handlers.PopulationSize)
           );		
    
    
    // Copy structure to GPU 
    cutilSafeCall( 
           cudaMemcpy(DeviceData, &FHost_Handlers, sizeof(TPopulationData),cudaMemcpyHostToDevice )
           );		
    
    
}// end of AllocateMemory
//------------------------------------------------------------------------------

/*
 * Free GPU memory
 */
void TGPU_Population::FreeCudaMemory(){
       
    
    // Free population data
    cutilSafeCall( 
           cudaFree(FHost_Handlers.Population)
           );  
     
    //Free Fitness data
    cutilSafeCall( 
           cudaFree(FHost_Handlers.Fitness) 
           );  
     
    
    // Free whole structure 
    cutilSafeCall( 
           cudaFree(DeviceData)
           );  
    
}// end of FreeMemory
//------------------------------------------------------------------------------
    
    
    
    
//----------------------------------------------------------------------------//
//                       TCPU_Population Implementation                       //
//                              public methods                                //
//----------------------------------------------------------------------------//

/*
 * Constructor of the class
 * @param PopulationSize
 * @param ChromosomeSize
 * 
 */
TCPU_Population::TCPU_Population(const int PopulationSize, const int ChromosomeSize){
  
    HostData = (TPopulationData *) malloc(sizeof (TPopulationData));
    HostData->ChromosomeSize = ChromosomeSize;
    HostData->PopulationSize = PopulationSize;
    
    AllocateCudaMemory();    
    
        
}// end of TCPU_Population
//------------------------------------------------------------------------------
    

/*
 * Destructor of the class
 */
TCPU_Population::~TCPU_Population(){
    
    FreeCudaMemory();
    
    free(HostData);
}// end of TCPU_Population
//------------------------------------------------------------------------------
    


/*
 * Print chromosome to string
 * 
 * @param Idx - Idx of chromosome in population
 */
string TCPU_Population::GetStringOfChromosome(const int Idx){
    
    
 stringstream  S;    
 
 // simple print of chromosome
 for (int BlockID=0; BlockID<HostData->ChromosomeSize; BlockID++){
     
     for (int BitID = 0; BitID < 32; BitID++ ) {
         char c = ((HostData->Population[Idx*HostData->ChromosomeSize + BlockID] & (1 << BitID)) == 0) ? '0' : '1';
         S << c;
     }    
         
     S << "\n";
     
  }
 
 
 return S.str();   
    
}// end of GetStringOfChromozome
//------------------------------------------------------------------------------



//----------------------------------------------------------------------------//
//                       TCPU_Population Implementation                       //
//                            protected methods                               //
//----------------------------------------------------------------------------//

/*
 * Allocate memory
 */
void TCPU_Population::AllocateCudaMemory(){
    
    // Allocate Population on the host side
    cutilSafeCall( 
            cudaHostAlloc((void**)&HostData->Population,  sizeof(TGene) * HostData->ChromosomeSize * HostData->PopulationSize,cudaHostAllocDefault )
            );		

    // Allocate fitness on the host side
    cutilSafeCall( 
            cudaHostAlloc((void**)&HostData->Fitness,  sizeof(TFitness) *  HostData->PopulationSize,cudaHostAllocDefault )
            );		
    
    
}// end of AllocateMemory
//------------------------------------------------------------------------------

/* 
 * Free memory
 */
void TCPU_Population::FreeCudaMemory(){
    
    // Free population on the host side
    cutilSafeCall( 
            cudaFreeHost(HostData->Population)
            );
    
    // Free fitness on the host side
    cutilSafeCall( 
            cudaFreeHost(HostData->Fitness)
            );            
    
}// end of FreeMemory
//------------------------------------------------------------------------------