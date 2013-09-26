/* 
 * File:        GlobalKnapsackData.h
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
 * Comments:    Header file of the knapsack global data class. 
 *              Data resides in GPU memory
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

#ifndef GLOBALKNAPSACKDATA_H
#define	GLOBALKNAPSACKDATA_H



typedef int TPriceType;
typedef int TWeightType;

/*
 * Global knapsack data structure accessible form CUDA kernels
 */
struct TKnapsackData{
    int           NumberOfItems;                // Number of items in knapsack
    int           OriginalNumberOfItems;        // Original size without padding to multiple of 32
    int           KnapsackCapacity;             // Total knapsack capacity
    float         MaxPriceWightRatio;           // What is the best Price-Weight ration (to penalization)
    
    TPriceType  * ItemPrice;                    // An array listing all item prices
    TWeightType * ItemWeight;                   // An array listing all item weights 
};// end of TKnapsackData
//------------------------------------------------------------------------------


/*
 * Global data for Knapsack Benchmark, class on the host side
 */
class TGlobalKnapsackData{        
public:    
    TKnapsackData * DeviceData; // Pointer to Device (GPU) data - necessary for transfers
    TKnapsackData * HostData;   // Host copy of global data (read from file)
    
    // Constructor of the class
    TGlobalKnapsackData();
    
    // Destructor of the class
    virtual ~TGlobalKnapsackData();
    
    
    // Load data from file
    void LoadFromFile();
        
protected:
    
    // Memory allocation and deallocation
    void AllocateMemory(int NumberOfItems);
    void FreeMemory();
    
    // Upload data to Device
    void UploadDataToDevice();
    
    // Handlers on inner arrays of TKnapsack Data on GPU (we cannot copy the structure at once)
    //   Because the size of following arrays not known in compiling time
    TPriceType  * FDeviceItemPriceHandler;
    TWeightType * FDeviceItemWeightHandler;            
}; // end of TGlobalKnapsackData
//------------------------------------------------------------------------------



#endif	/* GLOBALKNAPSACKDATA_H */

