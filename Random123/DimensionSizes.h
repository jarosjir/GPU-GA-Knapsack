/* 
 * File:   DimensionSizes.h
 * Author: jaros
 *
 * Created on 23 November 2011, 1:00 PM
 */

#ifndef DIMENSIONSIZES_H
#define	DIMENSIONSIZES_H

struct TDimensionSizes {
    size_t X; // X dimension size
    size_t Y; // Y dimension size
    size_t Z; // Z dimension size        
    TDimensionSizes(): X(0), Y(0), Z(0) {};
    TDimensionSizes(size_t x, size_t y, size_t z): X(x), Y(y), Z(z) {};
    
}; // end of TDimensionSizes
//------------------------------------------------------------------------------


#endif	/* DIMENSIONSIZES_H */

