# Implementação de paralelização com CUDA

Implementação paralelizada do algoritmo K-Means Naive utilizando a arquitetura CUDA da NVIDIA. O programa ```kmeans_1d_cuda.cu``` implementa um modelo de computação heterogênea (*Host/Device*), onde a CPU (*host*) orquestra o processo e a GPU (*device*) executa os cálculos intensivos. O fluxo do algoritmo lê os dados no *host*, copia o conjunto de pontos (X) para o *device* uma única vez e, a cada iteração, segue os passos do "Desenho Mínimo": Copia os centróides (C) do *host* para o *device* (H2D); Lança um *kernel* (*assignment_kernel*) onde cada *thread* da GPU processa um ponto de dado $i$, calculando o *assignment* e o erro (SSE) individual; Copia os *arrays* de *assign* e SSE individual de volta para o *host* (D2H); Executa a etapa de *update* (cálculo das médias) na CPU.

## Como compilar:

```bash

nvcc -O2 cuda/kmeans_cuda.cu -o cuda/kmeans_cuda -lm -Xcompiler -fopenmp
./cuda/kmeans_cuda data/dados.csv data/centroides_iniciais.csv 50 0.000001 cuda/assign.csv cuda/centroids.csv
cat cuda/centroids.csv

```