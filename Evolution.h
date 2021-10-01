/**
 * @file        Evolution.h
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
 * @brief       Header file of the GA evolution
 *              This class controls the evolution process on multicore CPU
 *
 * @date        30 March     2012, 00:00 (created)
 *              23 September 2021, 15:51 (revised)
 *
 * @copyright   Copyright (C) 2012 - 2021 Jiri Jaros.
 *
 * This source code is distribute under OpenSouce GNU GPL license.
 * If using this code, please consider citation of related papers
 * at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 */


#ifndef EVOLUTION_H
#define	EVOLUTION_H

#include "Parameters.h"
#include "Population.h"
#include "Statistics.h"
#include "GlobalKnapsackData.h"


/**
 * @class GPUEvolution
 * @brief Class controlling the evolutionary process.
 */
class GPUEvolution
{
  public:
    /// Class constructor.
    GPUEvolution();

    // Copy constructor not allowed.
    GPUEvolution(const GPUEvolution&) = delete;

    /// Destructor.
    virtual ~GPUEvolution();

    /// Run evolution.
    void run();

  protected:

    /// Initialize evolution.
    void initialize();

    /// Run evolution.
    void runEvolutionCycle();

    /// Init random generator seed.
    void initRandomSeed();
    /// Get random generator seed and increment it.
    unsigned int getRandomSeed() { return mRandomSeed++; };


    /// Parameters of evolution
    Parameters& mParams;
    /// Actual generation.
    int mActGeneration;
    /// Number of SM on GPU.
    int mMultiprocessorCount;
    /// Device Id.
    int mDeviceIdx;
    /// Random Generator Seed.
    unsigned int mRandomSeed;

    /// Master GA population
    GPUPopulation* mMasterPopulation;
    /// Population of offsprings
    GPUPopulation* mOffspringPopulation;

    /// Statistics over GA process
    GPUStatistics* mStatistics;

    /// Global data of knapsack
    GlobalKnapsackData mGlobalData;
};// end of GPU_Evolution
//----------------------------------------------------------------------------------------------------------------------

#endif	/* EVOLUTION_H */

