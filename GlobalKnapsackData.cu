/* 
 * File:        GlobalKnapsackData.cu
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
 * Comments:    Implementation file of the knapsack global data class. 
 *              This class maintains the benchmark data
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

#include <helper_cuda.h>
#include <fstream>
#include <iostream>

#include "GlobalKnapsackData.h"
#include "Parameters.h"


//----------------------------------------------------------------------------//
//                              Definitions                                   //
//----------------------------------------------------------------------------//


static const char * const ERROR_FILE_NOT_FOUND = "Global Benchmark Data: File not found";


//----------------------------------------------------------------------------//
//                              public methods                                //
//----------------------------------------------------------------------------//


/*
 * Destructor of the class 
 */   
TGlobalKnapsackData::~TGlobalKnapsackData(){
    
    FreeMemory();
    
}// end of TGlobalKnapsackData
//------------------------------------------------------------------------------
    


/*
 * Load data from file, filename given in Parameter class
 */
void TGlobalKnapsackData::LoadFromFile(){
        
    // Get instance of Parameter class
    TParameters * Params = TParameters::GetInstance();
            
    // Open file with benchmark data
    ifstream fr(Params->BenchmarkFileName().c_str());   
    if (!fr.is_open()) {
        cerr << ERROR_FILE_NOT_FOUND << endl; 
        exit(1);
    }
 
    
    // Read number of items
    int NumberOfItems = 0;    
    fr >>NumberOfItems;
    
    int OriginalNumberOfItems = NumberOfItems;
    
    // Calculate padding
    int Overhead = NumberOfItems % (Params->IntBlockSize() * WARP_SIZE);
    if (Overhead != 0) NumberOfItems = NumberOfItems + ((Params->IntBlockSize() * WARP_SIZE) - Overhead);
    
    // Allocate memory for arrays
    AllocateMemory(NumberOfItems);
        
    HostData->NumberOfItems         = NumberOfItems;
    HostData->OriginalNumberOfItems = OriginalNumberOfItems;
    
    
    
    // Load prices
    for (size_t i = 0; i < OriginalNumberOfItems; i++){
        fr >> HostData->ItemPrice[i];    
                                   
    }    // add padding
    for (size_t i = OriginalNumberOfItems; i < NumberOfItems; i++){
        HostData->ItemPrice[i] = TPriceType(0);                       
    }

    
    
    // Load weights 
    for (size_t i = 0; i < OriginalNumberOfItems; i++){
        fr >> HostData->ItemWeight[i];
    } // add padding
    for (size_t i = OriginalNumberOfItems; i < NumberOfItems; i++){
        HostData->ItemWeight[i] = TPriceType(0);        
    }
    
    
    // Get max Price/Weight ratio
    HostData->MaxPriceWightRatio = 0.0f;
    
    for (size_t i = 0; i < OriginalNumberOfItems; i++){
        if (HostData->ItemWeight[i] != 0) {
                float Ratio = HostData->ItemPrice[i] / HostData->ItemWeight[i];
                if (Ratio > HostData->MaxPriceWightRatio)  HostData->MaxPriceWightRatio = Ratio;
        }
        
    }
    
    //Read Knapsack capacity
    fr >> HostData->KnapsackCapacity;
    
    // Update chromosome size in parameters
    Params->SetChromosomeSize(NumberOfItems/Params->IntBlockSize());
    
    
    // Upload gloal data to device memory
    UploadDataToDevice();
        
    
}// end of LoadFromFile
//------------------------------------------------------------------------------
    
    
    
//----------------------------------------------------------------------------//
//                           protected methods                                //
//----------------------------------------------------------------------------//

/*
 * Allocate memory
 * 
 * @param       NumberOfItems - Number of Items in Knapsack with padding
 */
void TGlobalKnapsackData::AllocateMemory(int NumberOfItems){
    
    
    //------------------------- Host allocation ------------------------------//    
    //------------------- All data allocated by PINNED memory ----------------//
    checkCudaErrors(
           cudaHostAlloc((void**)&HostData,  sizeof(TKnapsackData), cudaHostAllocDefault)
           );		
    
    checkCudaErrors(
           cudaHostAlloc((void**)&HostData->ItemPrice,  sizeof(TPriceType) * NumberOfItems, cudaHostAllocDefault)
           );		
    
    checkCudaErrors(
           cudaHostAlloc((void**)&HostData->ItemWeight,  sizeof(TWeightType)* NumberOfItems, cudaHostAllocDefault)
           );		
    
    
    
    
    //----------------------- Device allocation ------------------------------//    
    checkCudaErrors(
           cudaMalloc((void**)&(DeviceData),  sizeof(TKnapsackData) )
           );		
            
    checkCudaErrors(
           cudaMalloc((void**)&(FDeviceItemPriceHandler),  sizeof(TPriceType) * NumberOfItems)
           );		
        
    checkCudaErrors(
           cudaMalloc((void**)&(FDeviceItemWeightHandler),  sizeof(TWeightType) * NumberOfItems)
           );		
            
    
}// end of AllocateMemory
//------------------------------------------------------------------------------


/*
 * Free Memory
 */
void TGlobalKnapsackData::FreeMemory(){
    
    //------------------------- Host allocation ------------------------------//        
    checkCudaErrors(
           cudaFreeHost(HostData->ItemPrice)
           );		
    
    checkCudaErrors(
           cudaFreeHost(HostData->ItemWeight)            
           );		
    
    checkCudaErrors(
           cudaFreeHost(HostData)
           );		
    
    
    
    
    //----------------------- Device allocation ------------------------------//    
    checkCudaErrors(
           cudaFree(DeviceData)
           );		
        
    checkCudaErrors(
           cudaFree(FDeviceItemPriceHandler)
           );		        
    checkCudaErrors(
           cudaFree(FDeviceItemWeightHandler)
           );		        
    
}// end of AllocateMemory
//------------------------------------------------------------------------------



/*
 * Upload Data to Device
 */
void TGlobalKnapsackData::UploadDataToDevice(){
    
    
    // Copy basic structure - struct data
    checkCudaErrors(
         cudaMemcpy(DeviceData, HostData, sizeof(TKnapsackData), 
                    cudaMemcpyHostToDevice)
         );    
        
    
    // Set pointer of the ItemPrice vector into the struct on GPU (link struct and vector)
    checkCudaErrors(
         cudaMemcpy(&(DeviceData->ItemPrice), &FDeviceItemPriceHandler, sizeof(TPriceType * ), 
                    cudaMemcpyHostToDevice)
         );    
    

    // Set pointer of the ItemWeight vector into struct on GPU (link struct and vector)
    checkCudaErrors(
         cudaMemcpy(&(DeviceData->ItemWeight), &FDeviceItemWeightHandler, sizeof(TWeightType * ), 
                    cudaMemcpyHostToDevice)
         );    
    
    
        
    // Copy prices 
    checkCudaErrors(
         cudaMemcpy(FDeviceItemPriceHandler, HostData->ItemPrice,  sizeof(TPriceType) * HostData->NumberOfItems, 
                    cudaMemcpyHostToDevice)
         );    
    
    // Copy weights 
    checkCudaErrors(
         cudaMemcpy(FDeviceItemWeightHandler, HostData->ItemWeight, sizeof(TWeightType) * HostData->NumberOfItems, 
                    cudaMemcpyHostToDevice)
         );    
  
    
}// end of UploadDataToDevice
//------------------------------------------------------------------------------