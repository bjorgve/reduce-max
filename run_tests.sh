#!/bin/sh

# compile the program
nvcc -o max-finder main-max.cu reduce-max.cu

# run the program with varying matrix sizes and maximum numbers
for i in {1..10}
do
   echo "Running test $i"
   ./max-finder $((1024*1024*$i)) $((100*$i))
   echo ""
done
