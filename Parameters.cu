/**
 * @file        Parameters.cu
 * @author      Jiri Jaros
 *              Brno University of Technology
 *              Faculty of Information Technology
 *
 *              and
 *
 *              The Australian National University
 *              ANU College of Engineering & Computer Science
 *
 *              jarosjir@fit.vutbr.cz
 *              www.fit.vutbr.cz/~jarosjir
 *
 * @brief       Implementation file of the parameter class.
 *              This class maintains all the parameters of evolution.
 *
 * @date        30 March     2012, 00:00 (created)
 *              22 September 2021: 18:59
 *
 * @copyright   Copyright (C) 2012 - 2021 Jiri Jaros.
 *
 * This source code is distribute under OpenSouce GNU GPL license.
 * If using this code, please consider citation of related papers
 * at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 */

#include <getopt.h>

#include <helper_cuda.h>
#include <cuda_runtime.h>

#include "Parameters.h"



//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------- Definitions ----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/// Copy of Evolutionary parameters in device constant memory.
__constant__  EvolutionParameters gpuEvolutionParameters;


// Singleton initialization
bool Parameters::sInstanceFlag = false;
Parameters* Parameters::sSingletonInstance = nullptr;


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Get instance of Parameters
 */
Parameters& Parameters::GetInstance()
{
  if(!sInstanceFlag)
  {
    sSingletonInstance = new Parameters();
    sInstanceFlag = true;
    return *sSingletonInstance;
  }
  else
  {
    return *sSingletonInstance;
  }
}// end of Parameters::GetInstance
//----------------------------------------------------------------------------------------------------------------------

/**
 * Load parameters from command line
 */
void Parameters::parseCommandline(int    argc,
                                  char** argv)
{
  // Default values
  float offspringPercentage = 0.5f;
  char c;

  // Parse command line
  while ((c = getopt (argc, argv, "p:g:m:c:o:f:s:bh")) != -1)
  {
    switch (c)
    {
      case 'p':
      {
        if (atoi(optarg) != 0)
        {
          mEvolutionParameters.populationSize = atoi(optarg);
        }
        break;
      }
      case 'g':
      {
        if (atoi(optarg) != 0)
        {
          mEvolutionParameters.numOfGenerations = atoi(optarg);
        }
        break;
      }

      case 'm':
      {
        if (atof(optarg) != 0)
        {
          mEvolutionParameters.mutationPst = atof(optarg);
        }
        break;
      }

      case 'c':
      {
        if (atof(optarg) != 0)
        {
          mEvolutionParameters.crossoverPst = atof(optarg);
        }
        break;
      }
      case 'o':
      {
        if (atof(optarg) != 0)
        {
          offspringPercentage = atof(optarg);;
        }
        break;
      }

      case 's':
      {
        if (atoi(optarg) != 0)
        {
          mEvolutionParameters.statisticsInterval = atoi(optarg);
        }
        break;
      }

      case 'b':
      {
        mPrintBest = true;
        break;
      }

      case 'f':
      {
        mGlobalDataFileName  = optarg;
        break;
      }
      case 'h':
      {
        printUsageAndExit();
        break;
      }

      default:
      {
        printUsageAndExit();
      }
     }
   }// switch

   // Set population size to be even.
   mEvolutionParameters.offspringPopulationSize = int(offspringPercentage * mEvolutionParameters.populationSize);
   if (mEvolutionParameters.offspringPopulationSize == 0)
   {
     mEvolutionParameters.offspringPopulationSize = 2;
   }
   if (mEvolutionParameters.offspringPopulationSize % 2 == 1)
   {
     mEvolutionParameters.offspringPopulationSize++;
   }


  // Set UINT mutation threshold to faster comparison
  mEvolutionParameters.mutationUintBoundary  = (unsigned int) ((float) UINT_MAX * mEvolutionParameters.mutationPst);
  mEvolutionParameters.crossoverUintBoundary = (unsigned int) ((float) UINT_MAX * mEvolutionParameters.crossoverPst);

} // end of parseCommandline
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy parameters to the GPU constant memory
 */
void Parameters::copyToDevice()
{
  checkCudaErrors(cudaMemcpyToSymbol(gpuEvolutionParameters, &mEvolutionParameters, sizeof(mEvolutionParameters)));

}// end of copyToDevice
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Private methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor
 */
Parameters::Parameters()
{
  mEvolutionParameters.populationSize      = 128;
  mEvolutionParameters.chromosomeSize      = 32;
  mEvolutionParameters.numOfGenerations    = 100;

  mEvolutionParameters.mutationPst         = 0.01f;
  mEvolutionParameters.crossoverPst        = 0.7f;
  mEvolutionParameters.offspringPopulationSize = (int) (0.5f * mEvolutionParameters.populationSize);

  mEvolutionParameters.statisticsInterval  = 1;

  mEvolutionParameters.intBlockSize        = sizeof(int) * 8;
  mGlobalDataFileName                      = "";

  mPrintBest                               = false;
}// end of Parameters
//----------------------------------------------------------------------------------------------------------------------

/**
 * print usage of the algorithm
 */
void Parameters::printUsageAndExit()
{
  fprintf(stderr, "Parameters for the genetic algorithm solving knapsack problem: \n");
  fprintf(stderr, "  -p population_size\n");
  fprintf(stderr, "  -g number_of_generations\n");
  fprintf(stderr, "\n");

  fprintf(stderr, "  -m mutation_rate\n");
  fprintf(stderr, "  -c crossover_rate\n");
  fprintf(stderr, "  -o offspring_rate\n");
  fprintf(stderr, "\n");

  fprintf(stderr, "  -s statistics_interval\n");
  fprintf(stderr, "\n");

  fprintf(stderr, "  -b print best individual\n");
  fprintf(stderr, "  -f benchmark_file_name\n");


  fprintf(stderr, "\n");
  fprintf(stderr, "Default population_size       = 128\n");
  fprintf(stderr, "Default number_of_generations = 100\n");
  fprintf(stderr, "\n");

  fprintf(stderr, "Default mutation_rate  = 0.01\n");
  fprintf(stderr, "Default crossover_rate = 0.7\n");
  fprintf(stderr, "Default offspring_rate = 0.5\n");
  fprintf(stderr, "\n");


  fprintf(stderr, "Default statistics_interval = 1\n");

  fprintf(stderr, "Default benchmark_file_name = knapsack_data.txt\n");

  exit(EXIT_FAILURE);
}// end of printUsage
//----------------------------------------------------------------------------------------------------------------------


/**
 * Print all parameters
 */
void Parameters::printOutAllParameters()
{
  printf("-----------------------------------------\n");
  printf("--- Evolution parameters --- \n");
  printf("Population size:     %d\n", mEvolutionParameters.populationSize);
  printf("Offspring size:      %d\n", mEvolutionParameters.offspringPopulationSize);
  printf("Chromosome int size: %d\n", mEvolutionParameters.chromosomeSize);
  printf("Chromosome size:     %d\n", mEvolutionParameters.chromosomeSize * mEvolutionParameters.intBlockSize);

  printf("Num of generations:  %d\n", mEvolutionParameters.numOfGenerations);
  printf("\n");


  printf("Crossover pst:       %f\n", mEvolutionParameters.crossoverPst);
  printf("Mutation  pst:       %f\n", mEvolutionParameters.mutationPst);
  printf("Crossover  int:      %u\n",mEvolutionParameters.crossoverUintBoundary);
  printf("Mutation  int:       %u\n", mEvolutionParameters.mutationUintBoundary);
  printf("\n");

  printf("Statistics interval: %d\n", mEvolutionParameters.statisticsInterval);

  printf("\n");
  printf("Data File: %s\n",mGlobalDataFileName.c_str());
  printf("-----------------------------------------\n");
}// end of PrintAllParameters
//----------------------------------------------------------------------------------------------------------------------