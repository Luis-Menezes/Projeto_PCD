# K-means 1D (*naive*) com Paralelização Progressiva

Este repositório consiste no projeto final da disciplina de Programação Concorrente e Distribuída, ministrada pelos professores Dr. Álvaro Fazenda e Denise Stringhini na Universidade Federal de São Paulo (UNIFESP). 

## Alunos:
Isabella Mariana Cardoso Pinto
- RA: 164915
- Integral
Luis Filipe Carvalho de Menezes
- RA: 164924
- Noturno

## Arquitetura do projeto:
O projeto consiste na análise progressiva da paralelização do algoritmo de *clusterização* **K-means 1D**. Dessa forma, inicialmente será feita a análise entre a implementação serial e com paralelização pela CPU com memória compartilhada utilizando OpenMP. Posteriormente, implementações com paralelização pela GPU e por memória distribuída (MPI). 

Cada abordagem do algoritmo será estará em uma pasta específica: ```serial/```, ```openMP/```, ```cuda/``` e ```mpi/```. A pasta ```data/``` contém o ```.csv``` que será utilizado pelo algoritmo em geral, bem como o código utilizado para gerar esses pontos.