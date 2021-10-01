/**
 * @file        main.cpp
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
 * @brief       Efficient GPU implementation of the Genetic Algorithm solving the Knapsack problem.
 *
 * @date        30 March     2012, 00:00 (created)
 *              22 September 2021, 17:29 (revised)
 *
 *
 * @mainpage  GPU accelerated genetic algorithm running the 0/1 knapsack problem
 *
 * @section   Usage
 *\verbatim
 *   -p population_size
 *   -g number_of_generations
 *
 *   -m mutation_rate
 *   -c crossover_rate
 *   -o offspring_rate
 *
 *   -s statistics_interval
 *
 *   -b print best individual
 *   -f benchmark_file_name
 *
 *   Default population_size       = 128
 *   Default number_of_generations = 100
 *
 *   Default mutation_rate         = 0.01
 *   Default crossover_rate        = 0.7
 *   Default offspring_rate        = 0.5
 *
 *   Default statistics_interval = 1
 *   Default benchmark_file_name = knapsack_data.txt
 *\endverbatim
 *
 * @copyright   Copyright (C) 2012 - 2021 Jiri Jaros.
 *
 * This source code is distribute under OpenSouce GNU GPL license.
 * If using this code, please consider citation of related papers
 * at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 */


#include <stdio.h>

#include "Evolution.h"
#include "Parameters.h"

/**
 * The main function
 */
int main(int argc, char **argv)
{
  // Load parameters
  Parameters& params = Parameters::GetInstance();
  params.parseCommandline(argc,argv);

  // Create GPU evolution class
  GPUEvolution GPU_Evolution;

  unsigned int AlgorithmStartTime;
  AlgorithmStartTime = clock();

  // Run evolution
  GPU_Evolution.run();

  unsigned int AlgorithmStopTime = clock();
  printf("Execution time: %0.3f s.\n", (float)(AlgorithmStopTime - AlgorithmStartTime) / (float) CLOCKS_PER_SEC);

  return EXIT_SUCCESS;
}// end of main
//----------------------------------------------------------------------------------------------------------------------
