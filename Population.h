/**
 * @file        Population.h
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
 * @brief       Header file of the GA population
 *              This class maintains and GA populations
 *
 * @date        30 March     2012, 00:00 (created)
 *              23 September 2021, 12:26 (revised)
 *
 * @copyright   Copyright (C) 2012 - 2021 Jiri Jaros.
 *
 * This source code is distribute under OpenSouce GNU GPL license.
 * If using this code, please consider citation of related papers
 * at http://www.fit.vutbr.cz/~jarosjir/pubs.php
 *
 */

#ifndef POPULATION_H
#define	POPULATION_H

#include <string>

/// Data type for Gene.
typedef unsigned int Gene;
/// Data type for fitness value
typedef float        Fitness;

/**
 * @struct PopulationData
 * @brief  Population data structure.
 */
struct PopulationData
{
  /// Number of chromosomes.
  unsigned int populationSize;
  /// Size of chromosome in INTs.
  unsigned int chromosomeSize;

  /// 1D array of genes (chromosome-based encoding).
  Gene*    population;
  /// 1D array of fitness values.
  Fitness* fitness;
};// end of PopulationData
//----------------------------------------------------------------------------------------------------------------------

/**
 * @class GPUPopulation
 * @brief Population stored on the GPU.
 */
class GPUPopulation
{
  public:
    /// Default constructor not allowed.
    GPUPopulation() = delete;
    /// Copy constructor not allowed.
    GPUPopulation(const GPUPopulation& orig) = delete;

    /**
     * Constructor.
     * @param [in] populationSize - Number of chromosomes.
     * @param [in] chromosomeSize - Chromosome length.
     */
    GPUPopulation(const int populationSize,
                  const int chromosomeSize);

    /// Destructor
    virtual ~GPUPopulation();
    /// Get pointer to device population data.
    PopulationData* getDeviceData()             { return mDeviceData; };
    /// Get pointer to device population data, const version.
    const PopulationData* getDeviceData() const { return mDeviceData; };

    /**
     * @brief Copy data from CPU population structure to GPU.
     * Both population must have the same size (sizes not being copied)!!
     *
     * @param [in] hostPopulatoin - Source of population data on the host side.
    */
    void copyToDevice(const PopulationData* hostPopulatoin);
    /**
     * @brief Copy data from GPU population structure to CPU.
     * Both population must have the same size (sizes not copied)!!
     *
     * @param [out] hostPopulatoin - Source of population data on the host side
     */
    void copyFromDevice(PopulationData* hostPopulatoin);
    /**
     * @brief Copy data from different population (both on the same GPU)
     * No size check!!!
     *
     * @param [in] sourceDevicePopulation - Source population.
     */
    void copyOnDevice(const GPUPopulation* sourceDevicePopulation);

    /**
     * Copy a given individual from device to host
     * @param [out] individual - Where to store an individual.
     * @param [in]  index      - Index of the individual in device population
     */
    void copyIndividualFromDevice(Gene* individual,
                                  int   index);

  protected:
    /// Allocate memory.
    void allocateMemory();
    /// Free memory.
    void freeMemory();

  private:
    /// Hander on the GPU data
    PopulationData* mDeviceData;

    /// Host copy of population
    PopulationData mHostPopulationHandler;

};// end of TGPU_Population
//----------------------------------------------------------------------------------------------------------------------


/**
 * @class CPUPopulation
 * @brief Population stored on the host side .
 *
 */
class CPUPopulation
{
  public:
    /// Default constructor not allowed.
    CPUPopulation() = delete;
    /// Default copy constructor not allowed.
    CPUPopulation(const CPUPopulation&) = delete;

    /**
     * Constructor
     * @param [in] populationSize - Number of chromosomes.
     * @param [in] chromosomeSize - Chromosome length.
     */
    CPUPopulation(const int populationSize,
                  const int chromosomeSize);

    /// Destructor
    virtual ~CPUPopulation();

    /// Get pointer to device population data.
    PopulationData* getDeviceData()             { return mHostData; };
    /// Get pointer to device population data, const version.
    const PopulationData* getDeviceData() const { return mHostData; };

  protected:
    /// Allocate memory
    void allocateMemory();
    /// Free memory
    void freeMemory();

  private:
    /// Host population data
    PopulationData* mHostData;
};// end of CPUPopulation
//----------------------------------------------------------------------------------------------------------------------

#endif	/* POPULATION_H */

