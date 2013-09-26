/* 
 * File:        GPU_Statistics.h
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
 * Comments:    Header file of the GA statistics
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
 * Created on 30 March 2012, 00:00 PM
 */

#ifndef GPU_STATISTICS_H
#define	GPU_STATISTICS_H


#include "Parameters.h"
#include "GPU_Population.h"
#include "GlobalKnapsackData.h"


/*
 * Statistics Structure
 */
struct TStatisticsData{
   TFitness MinFitness; // Minimum fitness value in population
   TFitness MaxFitness; // Maximum fitness value in population
   float    AvgFitness; // Mean fitness value in population
   float    Divergence; // Divergence in population
   int      IndexBest;  // Which individual is the best   
   
   TStatisticsData(){
        MinFitness = TFitness(0);
        MaxFitness = TFitness(0);;
        AvgFitness = 0.0f;
        Divergence = 0.0f;   
        IndexBest  = 0;                        
   };
     
};// end of TStatisticsData
//------------------------------------------------------------------------------



/*
 * GPU statistics class
 * 
 */
class TGPU_Statistics {
public:
    
    TStatisticsData * DeviceData;       // pointer to the GPU version fo statistics struct
    TStatisticsData * HostData;         // statistics in Host memory - necessary for printing
            
    TGene * HostBestIndividual;         // host copy of the best individual chromosome
    
    // Calculate statistics
    void   Calculate(TGPU_Population * Population, bool PrintBest);    
    
    // Get best individual in text form
    string GetBestIndividualStr(TKnapsackData * GlobalKnapsackData);
    
    TGPU_Statistics();
    virtual ~TGPU_Statistics();

protected:
    // Memory Allocation
    void AllocateCudaMemory();
    void FreeCudaMemory();

    // Initialise statistics structure
    void InitStatistics();
    
    // Copy statistics data from GPU memory down to host
    void CopyOut(TGPU_Population * Population, bool PrintBest);   // Device -> Host        
    
private:         
   
    TGPU_Statistics(const TGPU_Population& orig);

};// end of TGPU_Statistics
//------------------------------------------------------------------------------


#endif	/* GPU_STATISTICS_H */

