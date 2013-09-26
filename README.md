GA-GPU-Knapsack
===============

GPGPU implementation of the GA running the knapsack benchmark

Outline

This package contains an efficient GPU implementation of the Knapsack benchmark. 
You can compile it by typing:	make

If you want to run a simple demo, type:	make run

It is essential for you to set CUDA paths in GPU version of makefile to be able to compile it.
For more information visit: http://www.fit.vutbr.cz/~jarosjir/pubs.php?id=9830&shortname=1
and read the content of 
Jaros, J., Pospichal, P.: A Fair Comparison of Modern CPUs and GPUs Running the Genetic Algorithm under the Knapsack Benchmark, 
In: EvoStar 2012, Malaga, 2012, p. 10


GPU Requirements:
NVIDIA GTX 4XX series (architecture 2.0)

Software Requirements:
Compiler: 	g++-4.4 or newer
		nvcc-5.5 or newer 

