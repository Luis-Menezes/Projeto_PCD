# K-means 1D (*naive*) com Paralelização Progressiva

Este repositório consiste no projeto final da disciplina de Programação Concorrente e Distribuída, ministrada pelos professores Dr. Álvaro Fazenda e Dra. Denise Stringhini na Universidade Federal de São Paulo (UNIFESP). 

## Alunos:

- Isabella Mariana Cardoso Pinto - RA: 164915 - Integral
- Luis Filipe Carvalho de Menezes - RA: 164924 - Noturno

## Arquitetura do projeto:

O projeto consiste na análise progressiva da paralelização do algoritmo de clusterização **K-means 1D**. A análise teve como ponto de partida a implementação serial. A partir dela, foram desenvolvidas três etapas: (i) paralelização em CPU com memória compartilhada utilizando OpenMP; (ii) paralelização heterogênea entre CPU e GPU com CUDA; e (iii) paralelização com memória distribuída utilizando MPI.

Cada abordagem do algoritmo está contida em uma pasta específica: ```serial/```, ```openMP/```, ```cuda/``` e ```mpi/```. A pasta ```data/``` contém o ```.csv``` que será utilizado pelo algoritmo em geral, bem como o código utilizado para gerar esses pontos.

Contemplando todas as etapas descritas no enunciado da disciplina:
- Etapa 0 - Versão Sequencial (baseline)
- Etapa 1 - OpenMP (CPU)
- Etapa 2 - CUDA (GPU)
- Etapa 3 - MPI (distribuída)
- Etapa 4 - Entrega Final