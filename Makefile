# 
# File:        Makefile
# Author:      Jiri Jaros
# Affiliation: Brno University of Technology
#              Faculty of Information Technology
#              
#              and
# 
#              The Australian National University
#              ANU College of Engineering & Computer Science
#
# Email:       jarosjir@fit.vutbr.cz
# Web:         www.fit.vutbr.cz/~jarosjir
# 
# Comments:    Efficient GPU implementation of the Genetic Algorithm, 
#              solving the Knapsack problem.
#
# 
# License:     This source code is distribute under OpenSouce GNU GPL license
#                
#              If using this code, please consider citation of related papers
#              at http://www.fit.vutbr.cz/~jarosjir/pubs.php        
#      
#
# 
# Created on 24 March 2012, 00:00
# Last midification on 21 September 2021, 21:59
#


# Environment
CC=nvcc
CXX=nvcc

CUDA_ARCH = --generate-code arch=compute_50,code=sm_50 \
            --generate-code arch=compute_52,code=sm_52 \
            --generate-code arch=compute_53,code=sm_53 \
            --generate-code arch=compute_60,code=sm_60 \
            --generate-code arch=compute_61,code=sm_61 \
            --generate-code arch=compute_62,code=sm_62 \
            --generate-code arch=compute_70,code=sm_70 \
            --generate-code arch=compute_72,code=sm_72 \
            --generate-code arch=compute_75,code=sm_75 \
            --generate-code arch=compute_80,code=sm_80 \
            --generate-code arch=compute_86,code=sm_86


CXXFLAGS= -Xptxas=-v -m64 -O3  --device-c ${CUDA_ARCH}
TARGET=gpu_knapsack
LDFLAGS=

#----------------------------------------------------------------
# CHANGE PATHS to CUDA!!
#CXXINCLUDE=-I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc
CXXINCLUDE=-I${EBROOTCUDA}/include -I${EBROOTCUDA}/samples/common/inc


#----------------------------------------------------------------

all:		$(TARGET)	

$(TARGET):	main.o CUDA_Kernels.o GPU_Statistics.o Parameters.o GPU_Population.o GPU_Evolution.o GlobalKnapsackData.o
	$(CXX) $(LDFLAGS) main.o CUDA_Kernels.o GPU_Statistics.o Parameters.o GPU_Population.o GPU_Evolution.o GlobalKnapsackData.o -lm -o $@ $(LIBS) 



main.o : main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

%.o : %.cu
	$(CXX) $(CXXFLAGS) ${CXXINCLUDE} -c $<


# Clean Targets
clean: 
	/bin/rm -f *.o *.~ $(TARGET)

run:
	./gpu_knapsack -f ./Data/knap_40.txt -p 100 -g 10000 -s 10

	


