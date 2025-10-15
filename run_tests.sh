#!/bin/bash

echo "Gerando um teste para implementação serial e OpenMP"
echo "Compilando ambos os códigos"

# Compile serial version
gcc -O2 -std=c99 serial/kmeans_1d_serial.c -o serial/kmeans_1d_naive -lm

# Compile OpenMP version
gcc -O2 -std=c99 -fopenmp openMP/opMP.c -o openMP/kmeans_omp -lm

echo "Executando ambos os códigos com os mesmos dados de entrada"
echo "Implementação serial:"
./serial/kmeans_1d_naive data/dados.csv data/centroides_iniciais.csv 50 0.000001 serial/assign.csv serial/centroids.csv

echo ""
echo "Implementação OpenMP:"
./openMP/kmeans_omp data/dados.csv data/centroides_iniciais.csv 50 0.000001 openMP/assign.csv openMP/centroids.csv