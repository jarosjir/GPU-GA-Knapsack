/* 
 * File:        main.cpp 
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
 * Comments:    Efficient GPU implementation of the Genetic Algorithm, 
 *              solving the Knapsack problem.
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


#include <iostream>
#include <stdio.h>

#include "GPU_Evolution.h"
#include "Parameters.h"



using namespace std;

/*
 * The main function
 */
int main(int argc, char **argv)
{
    
      // Load parameters
    TParameters * Params = TParameters::GetInstance();
    Params->LoadParametersFromCommandLine(argc,argv);
    
      // Create GPU evolution class
    TGPU_Evolution GPU_Evolution;    
    
    unsigned int AlgorithmStartTime;
    AlgorithmStartTime = clock();
    
      // Run evolution
    GPU_Evolution.Run();
    
    unsigned int AlgorithmStopTime = clock();        
    printf("Execution time: %0.3f s.\n",  (float)(AlgorithmStopTime - AlgorithmStartTime) / (float)CLOCKS_PER_SEC);    
    
    return 0;
}// end of main
//------------------------------------------------------------------------------
