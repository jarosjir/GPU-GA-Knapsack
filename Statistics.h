/**
 * @file        Statistics.h
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
 * @brief       Header file of the GA statistics
 *              This class maintains and collects GA statistics
 *
 * @date        30 March     2012, 00:00 (created)
 *              22 September 2021, 19:59 (revised)
 *
 * @copyright   Copyright (C) 2012 - 2021 Jiri Jaros.
 *
 * This source code is distribute under OpenSouce GNU GPL license.
 * If using this code, please consider citation of related papers
 * at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 */

#ifndef STATISTICS_H
#define	STATISTICS_H


#include "Parameters.h"
#include "Population.h"
#include "GlobalKnapsackData.h"


/**
 * @struct StatisticsData
 * @brief  Statistics Structure
 */
struct StatisticsData
{
  /// Minimum fitness value in population.
  Fitness minFitness = Fitness(0);
  /// Maximum fitness value in population.
  Fitness maxFitness = Fitness(0);
  /// Mean fitness value in population.
  float   avgFitness = 0.0f;
  /// Divergence in population
  float   divergence = 0.0f;
  /// Which individual is the best
  int     indexBest  = 0;
};// end of StatisticsData
//----------------------------------------------------------------------------------------------------------------------


/**
 * @class GPUStatistics
 * @brief GPU statistics class.
 */
class GPUStatistics
{
  public:
    /// Constructor.
    GPUStatistics();

    /// Copy constructor is not allowed.
    GPUStatistics(const GPUStatistics& ) = delete;
    /// Destructor.
    virtual ~GPUStatistics();

    /**
     * Calculate statistics
     * @param [in] population - Population to calculate statistics of.
     * @param [in] printBest  - do we need to download the best individual to print.
     */
    void calculate(GPUPopulation* population,
                   bool           printBest);

    /**
     * Get best individual in text form.
     * @param  [in] globalKnapsackData  - Global knapsack data
     * @return String representation of the best individual.
     */
    std::string getBestIndividualStr(const KnapsackData* globalKnapsackData) const;

    /// Get host version of statistical data.
    StatisticsData*       getHostData()         { return  hostData; };
    /// Get host version of statistical data, const version.
    const StatisticsData* getHostData()   const { return  hostData; };

    /// Get device version of statistical data.
    StatisticsData*       getDeviceData()       { return  deviceData; };
    /// Get device version of statistical data, const version.
    const StatisticsData* getDeviceData() const { return  deviceData; };

  protected:
    /// Allocate memory.
    void allocateMemory();
    /// Free  memory.
    void freeMemory();

    /// Initialise statistics structure.
    void initStatistics();

    /**
     * Copy statistics data from GPU memory down to host.
     * @param [in] population - Population to calculate statistics of.
     * @param [in] printBest  - do we need to download the best individual to print.
     */
    void copyFromDevice(GPUPopulation* population,
                        bool             printBest);

  private:
    // Pointer to the GPU version of statistics struct.
    StatisticsData* deviceData;
    // Statistics in Host memory - necessary for printing.
    StatisticsData* hostData;
    /// Host copy of the best individual chromosome
    Gene* hostBestIndividual;
};// end of TGPU_Statistics
//----------------------------------------------------------------------------------------------------------------------


#endif	/* GPU_STATISTICS_H */

