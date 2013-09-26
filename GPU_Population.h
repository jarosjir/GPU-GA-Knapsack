/* 
 * File:        GPU_Population.h
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
 * Comments:    Header file of the GA population
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
 * Created on 30 March 2012, 00:00 PM
 */

#ifndef TGPU_POPULATION_H
#define	TGPU_POPULATION_H

#include <string>

using namespace std;

// Basic types
typedef unsigned int TGene;
typedef float        TFitness;



/*
 * Population data structure
 */
struct TPopulationData{
    unsigned int PopulationSize;        // Number of chromosomes
    unsigned int ChromosomeSize;        // Size of chromosome in INTs
    
    TGene    * Population;              // 1D array of genes (chromosome-based encoding)
    TFitness * Fitness;                 // 1D array of fitness values
};// end of TPopulationData
//------------------------------------------------------------------------------




/*
 * GPU population 
 * 
 */
class TGPU_Population{
public:
    TPopulationData * DeviceData;       // Hander on the GPU data 
    
    
    // Constructor
    TGPU_Population(const int PopulationSize, const int ChromosomeSize);

    // Copy data  Host   -> Device
    void CopyIn      (const TPopulationData * HostSource);        
    // Copy data Device -> Host        
    void CopyOut     (TPopulationData       * HostDestination);   
    // Copy data Device -> Device;
    void CopyDeviceIn(const TGPU_Population * GPUPopulation);     
    
    // Copy an individal down to host
    void CopyOutIndividual(TGene * Individual, int Index);
    
    
    // Destructor
    virtual ~TGPU_Population();
protected:
    // Memory allocation
    void AllocateCudaMemory();
    void FreeCudaMemory();
    
private:
    
    // Host copy of population
    TPopulationData FHost_Handlers;
                
    TGPU_Population();
    TGPU_Population(const TGPU_Population& orig);
        
};// end of TGPU_Population
//------------------------------------------------------------------------------







/*
 * CPU Population
 * 
 */
class TCPU_Population{
public:
    TPopulationData * HostData;
        
    TCPU_Population(const int PopulationSize, const int ChromosomeSize);
    
    
    string GetStringOfChromosome(const int Idx);
    
    virtual ~TCPU_Population();

protected:
    
    void AllocateCudaMemory();
    void FreeCudaMemory();
        
    
private:
    TCPU_Population();            
    TCPU_Population(const TCPU_Population& orig);
        
};// end of TCPU_Population
//------------------------------------------------------------------------------

#endif	/* TGPU_POPULATION_H */

