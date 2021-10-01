/**
 * @file        GlobalKnapsackData.h
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
 * @brief       Header file of the knapsack global data class.
 *              Data resides in GPU memory
 *              This class maintains the benchmark data
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

#ifndef GLOBAL_KNAPSACK_DATA_H
#define	GLOBAL_KNAPSACK_DATA_H

/// Data type for item prices.
typedef int PriceType;
/// Data type for item weights.
typedef int WeightType;

/**
 * @struct KnapsackData
 * @brief  Global knapsack data structure accessible form CUDA kernels
 */
struct KnapsackData
{
  /// Number of items in knapsack padded to nearest multiply of 32.
  int         numberOfItems;
  /// Original size without padding to multiple of 32.
  int         originalNumberOfItems;
  /// Total knapsack capacity
  int         knapsackCapacity;
  /// What is the best Price-Weight ratio (to penalization)
  float       maxPriceWightRatio;

  /// Array listing all item prices.
  PriceType * itemPrice;
  /// Array listing all item weights.
  WeightType* itemWeight;
};// end of KnapsackData
//----------------------------------------------------------------------------------------------------------------------


/**
 * @class GlobalKnapsackData
 * @brief Global data for Knapsack Benchmark, class on the host side.
 */
class GlobalKnapsackData
{
  public:
    /// Constructor of the class.
    GlobalKnapsackData() : mDeviceData(nullptr), mHostData(nullptr), mDeviceItemPriceHandler(nullptr),
                           mDeviceItemWeightHandler(nullptr) {};

    /// Destructor of the class.
    virtual ~GlobalKnapsackData();

    /// Load data from file.
    void LoadFromFile();

    /// Get knapsack data on device.
    KnapsackData* getDeviceData()             { return mDeviceData;};
    /// Get knapsack data on device, const version.
    const KnapsackData* getDeviceData() const { return mDeviceData;};

    /// Get knapsack data on host.
    KnapsackData* getHostData()               { return mHostData;};
    /// Get knapsack data on host, const version.
    const KnapsackData* getHostData()   const { return mHostData;};

  protected:

    /**
     * Allocate memory.
     * @param [in] numberOfItems - Number of Items in Knapsack with padding
     */
    void allocateMemory(int numberOfItems);
    /// Free memory.
    void freeMemory();

    /// Upload data to device.
    void copyToDevice();

    /// Pointer to Device (GPU) data - necessary for transfers.
    KnapsackData* mDeviceData;
    /// Host copy of global data (read from file).
    KnapsackData* mHostData;

    // Handlers on inner arrays of Knapsack Data on GPU (we cannot copy the structure at once)
    // Because the size of following arrays not known in compiling time
    /// Pointer to device price array.
    PriceType*  mDeviceItemPriceHandler;
    /// Pointer to weight price array.
    WeightType* mDeviceItemWeightHandler;
}; // end of TGlobalKnapsackData
//------------------------------------------------------------------------------

#endif	/* GLOBAL_KNAPSACK_DATA_H */

