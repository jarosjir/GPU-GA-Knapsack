/* 
 * File:        Parameters.h
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
 * Comments:    Header file of the parameter class. 
 *              This class maintains all the parameters of evolution.
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


#ifndef TPARAMETERS_H
#define	TPARAMETERS_H

#include <string>

using namespace std;

//----------------------------------------------------------------------------//
//------------------------- CUDA constants -----------------------------------//
//----------------------------------------------------------------------------//

#define BLOCK_SIZE (256)
#define WARP_SIZE  (32)
#define CHR_PER_BLOCK (BLOCK_SIZE/WARP_SIZE)


/*
 * Evolution Parameters structure
 */
struct TEvolutionParameters{
    int PopulationSize;                 // Population size
    int OffspringPopulationSize;        // Offspring population size
    int ChromosomeSize;                 // Length of binary chromosome in int chunks
    int NumOfGenerations;               // Total number of generations to evolve
        
    float CrossoverPst;                 // Crossover rate (as flaot)
    float MutationPst;                  // Mutaion rate   (as float      
    unsigned int CrossoverUINTBoundary; // Crossover rate as uint 
    unsigned int MutationUINTBoundary;  // Mutation rate as uint 
    
    int EmigrantCount;                  // Number of migrating individuals
    int MigrationInterval;              // Number of CPU threads
    int IslandCount;                    // Number of independent islands
    int StatisticsInterval;             // How often to print statistics
    
    int IntBlockSize;                   // size of int block (32 bin genes)
};// end of TEvolutionParameters
//------------------------------------------------------------------------------





/*
 * Singleton class with Parameters maitaining them in CPU and GPU constant memory
 */
class TParameters {
public:
    
    // Get instance of the singleton class
    static TParameters* GetInstance();
    
    // Destructor
    virtual ~TParameters() {
        pTParametersInstanceFlag = false;
    };
    
    // Parse command line and populate the class
    void LoadParametersFromCommandLine(int argc, char **argv);
       
    // Store GA paremetrs in GPU cosntant memory
    void StoreParamsOnGPU();
    
    
    
    // Return values of basic parameters
    int   PopulationSize()        const {return EvolutionParameters.PopulationSize; };
    int   ChromosomeSize()        const {return EvolutionParameters.ChromosomeSize; };
    void  SetChromosomeSize(unsigned int Value) { EvolutionParameters.ChromosomeSize = Value; };
    
    int   NumOfGenerations()      const {return EvolutionParameters.NumOfGenerations; };
    
    
    float        CrossoverPst()          const {return EvolutionParameters.CrossoverPst; };
    float        MutationPst()           const {return EvolutionParameters.MutationPst; };
    unsigned int CrossoverUINTBoundary() const {return EvolutionParameters.CrossoverUINTBoundary; };    
    unsigned int MutationUINTBoundary()  const {return EvolutionParameters.MutationUINTBoundary; };
    
    int  OffspringPopulationSize()       const {return EvolutionParameters.OffspringPopulationSize; };
    
    int  EmigrantCount()        const {return EvolutionParameters.EmigrantCount; };
    int  MigrationInterval()    const {return EvolutionParameters.MigrationInterval; };
    int  IslandCount()          const {return EvolutionParameters.IslandCount; };
    int  StatisticsInterval()   const {return EvolutionParameters.StatisticsInterval;};    
        
    int  IntBlockSize()         const {return EvolutionParameters.IntBlockSize;};  
    
    
    // Get filename with global data
    string BenchmarkFileName()          const {return GlobalDataFileName;};
    
    // Get number of SM processors on the GPU
    int  GetGPU_SM_Count()              const {return FGPU_SM_Count;};
    // Set number of SM processors on the GPU
    void SetGPU_SM_Count(int SM_Count) {FGPU_SM_Count = SM_Count;};
    
    // Print best solution?
    bool GetPrintBest()                 const {return FPrintBest;};
    
    // print parameters to stdout
    void PrintAllParameters();
    
private:        
    TEvolutionParameters EvolutionParameters;   
    string GlobalDataFileName;
        
    
    static bool pTParametersInstanceFlag;
    static TParameters *pTParametersSingle;
    
    int  FGPU_SM_Count;
    bool FPrintBest;      
    
    // print error message end exit if parameters are wrong
    void PrintUsageAndExit();
    
    TParameters();

    //Prevent copy-construction
    TParameters(const TParameters&);

    //Prevent assignment
    TParameters& operator=(const TParameters&);
    
};// end of TParameters
//------------------------------------------------------------------------------


#endif	/* TPARAMETERS_H */

