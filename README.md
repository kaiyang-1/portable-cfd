# portable-cfd
Accelarate computational fluid dynamics with ISO C++ parallel algorithms.

## Environment
Ubuntu 22.04.2

NVIDIA HPC SDK 23.3

## Build
parallel on CPU with TBB
```
g++ -std=c++20 -Ofast -ltbb -march=native -DNDEBUG -o heat.out diffusion/heat.cpp 
```
parallel on CPU with nvc++ compiler
```
nvc++ -stdpar=multicore -std=c++20 -O4 -fast -march=native -Mllvm-fast -DNDEBUG -o heat.out diffusion/heat.cpp
```
parallel on GPU with nvc++ compiler
```
nvc++ -stdpar=gpu -std=c++20 -O4 -fast -march=native -Mllvm-fast -DNDEBUG -o heat.out diffusion/heat.cpp
```
