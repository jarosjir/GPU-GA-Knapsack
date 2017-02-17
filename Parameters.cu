/* 
 * File:        Parameters.cu
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
 * Comments:    Implementation file of the parameter class. 
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
 * Modified on 17 February 2017, 15:59
 */


#include <iostream>
#include <getopt.h>

#include <helper_cuda.h>
#include <cuda_runtime.h>

#include "Parameters.h"

//----------------------------------------------------------------------------//
//                              Definitions                                   //
//----------------------------------------------------------------------------//

__constant__  TEvolutionParameters GPU_EvolutionParameters;


// Singleton initialization 
bool TParameters::pTParametersInstanceFlag = false;
TParameters* TParameters::pTParametersSingle = NULL;


//----------------------------------------------------------------------------//
//                              Implementation                                //
//                              public methods                                //
//----------------------------------------------------------------------------//

/*
 * Get instance of TPrarams
 */
TParameters* TParameters::GetInstance(){
    if(! pTParametersInstanceFlag)
    {        
        pTParametersSingle = new TParameters();
        pTParametersInstanceFlag = true;
        return pTParametersSingle;
    }
    else
    {
        return pTParametersSingle;
    }
}// end of TParameters::GetInstance
//-----------------------------------------------------------------------------


/*
 * Load parameters from command line
 * 
 * @param argc
 * @param argv
 * 
 */
void TParameters::LoadParametersFromCommandLine(int argc, char **argv){
    
   // default values
   float OffspringPercentage = 0.5f;
   char c;

   // Parse command line 
   while ((c = getopt (argc, argv, "p:g:m:c:o:i:n:f:s:bh")) != -1){
       switch (c){
          case 'p':{              
              if (atoi(optarg) != 0) EvolutionParameters.PopulationSize = atoi(optarg);
              break;
          }
          case 'g': {
              if (atoi(optarg) != 0) EvolutionParameters.NumOfGenerations = atoi(optarg);
              break;
          }
  
          
          case 'm': {
              if (atof(optarg) != 0) EvolutionParameters.MutationPst = atof(optarg);              
              break;
          }
          case 'c': {
              if (atof(optarg) != 0) EvolutionParameters.CrossoverPst = atof(optarg);
              break;
          }
          case 'o': {
              if (atof(optarg) != 0) OffspringPercentage = atof(optarg);;
              break;
          }
         
          
         case 'i': {
              if (atoi(optarg) != 0) EvolutionParameters.IslandCount = atoi(optarg);;
              break;
          }
        
          case 'n': {
              if (atoi(optarg) != 0) EvolutionParameters.MigrationInterval = atoi(optarg);
              break;
          }
          
         case 's': {
              if (atoi(optarg) != 0) EvolutionParameters.StatisticsInterval = atoi(optarg);
              break;
          }

         case 'b': {
              FPrintBest = true;
              break;
          }
          
         case 'f': {
              GlobalDataFileName  = optarg;
              break;
          }
          case 'h':{

             PrintUsageAndExit();
             break;        
          }
          default:{

               PrintUsageAndExit();
          }
       }    
   }   
   
   // Set population size to be even.
   EvolutionParameters.OffspringPopulationSize = (int) (OffspringPercentage * EvolutionParameters.PopulationSize);
   if (EvolutionParameters.OffspringPopulationSize == 0) EvolutionParameters.OffspringPopulationSize = 2;
   if (EvolutionParameters.OffspringPopulationSize % 2 == 1) EvolutionParameters.OffspringPopulationSize++;
   
      
   // Set UINT mutation threshold to faster comparison
   EvolutionParameters.MutationUINTBoundary  = (unsigned int) ((float) UINT_MAX * EvolutionParameters.MutationPst);
   EvolutionParameters.CrossoverUINTBoundary = (unsigned int) ((float) UINT_MAX * EvolutionParameters.CrossoverPst);
   
} // end of LoadParametersFromCommandLine
//------------------------------------------------------------------------------


/*
 * Copy parameters to the GPU constant memory
 */
void TParameters::StoreParamsOnGPU(){
            
    checkCudaErrors(
        cudaMemcpyToSymbol(GPU_EvolutionParameters, &EvolutionParameters, sizeof(TEvolutionParameters) )
    );
    
   
}// end of StoreParamsOnGPU
//------------------------------------------------------------------------------


//----------------------------------------------------------------------------//
//                              Implementation                                //
//                              private methods                               //
//----------------------------------------------------------------------------//

/*
 * Constructor
 */
TParameters::TParameters(){
    
    EvolutionParameters.PopulationSize      = 128;
    EvolutionParameters.ChromosomeSize      = 32;
    EvolutionParameters.NumOfGenerations    = 100;
        
    EvolutionParameters.MutationPst         = 0.01f;
    EvolutionParameters.CrossoverPst        = 0.7f;    
    EvolutionParameters.OffspringPopulationSize = (int) (0.5f * EvolutionParameters.PopulationSize);
    
    EvolutionParameters.IslandCount         = 1;
    EvolutionParameters.EmigrantCount       = 0;
    EvolutionParameters.MigrationInterval   = 0;
    EvolutionParameters.StatisticsInterval  = 1;
    
    EvolutionParameters.IntBlockSize        = sizeof(int)*8;  
    GlobalDataFileName                      = "";
    
    FPrintBest                              = false;
    
}// end of TParameters
//------------------------------------------------------------------------------

/*
 * print usage of the algorithm
 */
void TParameters::PrintUsageAndExit(){
    
  cerr << "Usage: " << endl;  
  cerr << "  -p Population_size\n";
  cerr << "  -g Number_of_generations\n";
  cerr << endl;
  
  cerr << "  -m mutation_rate\n";
  cerr << "  -c crossover_rate\n";
  cerr << "  -o offspring_rate\n";
  cerr << endl;
  
  cerr << "  -i island_count\n";
  cerr << "  -e emigrants_rate\n";
  cerr << "  -n migration_interval\n";
  cerr << "  -s statistics_interval\n";
  cerr << endl;
  
  cerr << "  -b print best individual\n";
  cerr << "  -f benchmark_file_name\n";
  
          
  cerr << endl;
  cerr << "Default Population_size       = 128"  << endl;
  cerr << "Default Number_of_generations = 100" << endl;
  cerr << endl;
  
  cerr << "Default mutation_rate  = 0.01" << endl;
  cerr << "Default crossover_rate = 0.7" << endl;
  cerr << "Default offspring_rate = 0.5" << endl;
  cerr << endl;
  
  cerr << "Default island_count        = 1"   << endl;
  cerr << "Default emigrants_rate      = 0.1" << endl;
  cerr << "Default migration_interval  = 0"   << endl;
  cerr << "Default statistics_interval = 1"   << endl;
  
  cerr << "Default benchmark_file_name = knapsack_data.txt\n";
  
  exit(1);
    
}// end of PrintUsage
//------------------------------------------------------------------------------





/*
 * Print all parameters
 * 
 */
void TParameters::PrintAllParameters(){
    printf("-----------------------------------------\n");
    printf("--- Evolution parameters --- \n");
    printf("Population size:     %d\n", EvolutionParameters.PopulationSize);
    printf("Offspring size:      %d\n", EvolutionParameters.OffspringPopulationSize);
    printf("Chromosome int size: %d\n", EvolutionParameters.ChromosomeSize);
    printf("Chromosome size:     %d\n", EvolutionParameters.ChromosomeSize * EvolutionParameters.IntBlockSize);
    
    printf("Num of generations:  %d\n", EvolutionParameters.NumOfGenerations);
    printf("\n");
        
    
    printf("Crossover pst:       %f\n", EvolutionParameters.CrossoverPst);
    printf("Mutation  pst:       %f\n", EvolutionParameters.MutationPst);
    printf("Crossover  int:      %u\n",EvolutionParameters.CrossoverUINTBoundary);    
    printf("Mutation  int:       %u\n", EvolutionParameters.MutationUINTBoundary);    
    printf("\n");
    
    printf("Emigrant count:      %d\n", EvolutionParameters.EmigrantCount);
    printf("Migration interval:  %d\n", EvolutionParameters.MigrationInterval);
    printf("Island count:        %d\n", EvolutionParameters.IslandCount);    
    printf("Statistics interval: %d\n", EvolutionParameters.StatisticsInterval);
    
    printf("\n");
    printf("Data File: %s\n",GlobalDataFileName.c_str());
    printf("-----------------------------------------\n");
    
    
}// end of PrintAllParameters
//------------------------------------------------------------------------------