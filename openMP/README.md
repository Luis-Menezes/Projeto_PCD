# Implementação de paralelização com OpenMP

A implementação paralelizada do algoritmo K-Means Naive. O programa ```opMP.c``` lê pontos e centróides iniciais de CSVs de uma coluna, itera os passos de *assignment* (atribuição pelo menor quadrado da distância) e *update* (média dos pontos por cluster) até atingir um número máximo de iterações ou uma variação relativa do SSE abaixo de *eps*. O código usa arrays C, X e assign simples em memória, trata clusters vazios copiando o primeiro ponto (estratégia simples) e mede tempo de execução, o que a torna um bom baseline para testes e validação. Por ser *single‑threaded* e de complexidade $O(N·K·iter)$, não é ideal para conjuntos muito grandes, mas é clara, fácil de entender e serve como ponto de partida para as otimizações realizadas nesse projeto.

Na execução do projeto foi realizado dois arquivos de código diferentes, um analisando ã variação no número de *threads* fornecido pelo usuário e outro analisando o tipo de scheduling que seria utilizado (*dynamic* ou *static*) e o tamanho dos chunks consumidos por cada *thread*.

## Como compilar:

```bash

gcc -O2 -std=c99 -fopenmp openMP/opMP.c -o openMP/kmeans_omp -lm
./openMP/kmeans_omp data/dados.csv data/centroides_iniciais.csv 50 0.000001 openMP/assign.csv openMP/centroids.csv
cat openMP/centroids.csv

```