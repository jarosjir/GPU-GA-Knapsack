/* 
 * File:        GPU_Evolution.h
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
 * Comments:    Header file of the GA evolution
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
 * Created on 24 March 2012, 00:00 PM
 */

#ifndef TGPU_EVOLUTION_H
#define	TGPU_EVOLUTION_H

#include "Parameters.h"
#include "GPU_Population.h"
#include "GPU_Statistics.h"
#include "GlobalKnapsackData.h"


/*
 * struct of seed
 */
struct r123_seed{
    unsigned long seed1;
    unsigned long seed2;
};// end of r123_seed
//------------------------------------------------------------------------------

/*
 * CPU evolution process
 * 
 */
class TGPU_Evolution{
public:
    // Class constructor
    TGPU_Evolution();
    
    // Run evolution
    virtual ~TGPU_Evolution();
    
    // Run evolution
    void Run(); 
    
protected:    
    TParameters * Params;                       // Parameters of evolution
    int pActGeneration;                         // Actual generation            
    int pMultiprocessorCount;                   // Number of SM on GPU
    int pDeviceIdx;                             // Device ID    
    unsigned int pSeed;                         // Random Generator Seed
    
    TGPU_Population* MasterPopulation;          // Master GA population  
    TGPU_Population* OffspringPopulation;       // Population of offsprings
    
    TGPU_Population* MigrationPopulation_In;    // Population of Immigrants
    TGPU_Population* MigrationPopulation_Out;   // Population of Emigrants
        
    TGPU_Statistics * GPUStatistics;            // Statistics over GA process    
    
    TGlobalKnapsackData GlobalData;             // Global data of knapsack
        
    
    // Initialize evolution
    void Initialize();   
    
    // Run evolution
    void RunEvolutionCycle();
    
    void InitSeed();
    
    unsigned int GetSeed() {return pSeed++; };
    
    // Copy constructor not allowed
    TGPU_Evolution(const TGPU_Evolution& orig);
};

#endif	/* TGPU_EVOLUTION_H */

