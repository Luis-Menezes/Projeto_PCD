# Implementação de paralelização com MPI

Implementação paralelizada do algoritmo K-Means Naive utilizando a biblioteca MPI (*Message Passing Interface*). O programa ```kmeans_1d_mpi.c``` implementa um modelo de computação distribuída baseado em troca de mensagens, onde múltiplos processos cooperam para executar o algoritmo de forma paralela. O fluxo do algoritmo segue uma estratégia de decomposição de dados (*data parallelism*), onde cada processo trabalha em uma partição local do conjunto de dados. Na inicialização, o processo raiz (*rank 0*) lê o conjunto completo de pontos (X) e os centróides iniciais (C), e distribui os dados entre todos os processos usando *MPI_Scatter*. A cada iteração, segue-se os passos: Broadcast dos centróides atuais para todos os processos (*MPI_Bcast*); Cada processo executa localmente a etapa de *assignment*, calculando as distâncias de seus pontos locais aos centróides e acumulando somas e contagens por cluster; Redução global (*MPI_Allreduce*) das somas e contagens parciais de todos os processos; O processo raiz calcula os novos centróides (etapa *update*) usando os dados agregados e verifica a convergência. Ao final, o processo raiz coleta os *assignments* de todos os processos (*MPI_Gather*) e salva os resultados.

## Como compilar:

```bash
mpicc -O2 mpi/kmeans_mpi.c -o mpi/kmeans_mpi -lm -fopenmp
mpirun -np 4 mpi/kmeans_mpi data/dados.csv data/centroides_iniciais.csv 50 0.000001 mpi/assign.csv mpi/centroids.csv
cat mpi/centroids.csv
```

<!-- ## Características da implementação:

- **Decomposição de dados**: Cada processo trabalha em uma partição local do dataset
- **Comunicação coletiva**: Uso de *MPI_Bcast*, *MPI_Allreduce* e *MPI_Gather* para sincronização
- **Balanceamento de carga**: Distribuição uniforme dos pontos entre processos
- **Escalabilidade**: Suporta execução com número variável de processos (1 a N)
- **Validação**: Garante resultados idênticos independente do número de processos -->