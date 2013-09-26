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
# Created on 24 March 2012, 00:00 PM
#


# Environment
CC=nvcc
CXX=nvcc
CXXFLAGS= -arch=sm_20 -Xptxas=-v -m64 -O3 
TARGET=gpu_knapsack
LDFLAGS=

#----------------------------------------------------------------
# CHANGE PATHS to CUDA!!
CXXINCLUDE=-I/usr/local/NVIDIA_GPU_Computing_SDK/shared/inc -I/usr/local/NVIDIA_GPU_Computing_SDK/C/common/inc/
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
	./gpu_knapsack -f ./Data/knap_40.txt -p 100 -g 50 -s 1

	


