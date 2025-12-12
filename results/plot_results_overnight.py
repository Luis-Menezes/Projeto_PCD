import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cole o conteúdo COMPLETO do seu arquivo txt aqui
raw_data = """
=== RELATÓRIO FINAL DE EXPERIMENTOS PCD ===
Inicio: Thu Dec 11 06:57:30 PM -03 2025
Máquina: cidag003
CPU:  Intel(R) Core(TM) i9-14900KF
===========================================

>>> [Fase 1] Compilando códigos...
Compilação finalizada.

==================================================================
DATASET: N=10000 pontos, K=4 clusters
==================================================================
-> Gerando arquivos de entrada...
-> [SERIAL] Executando Baseline...
K-means 1D (naive)
N=10000 K=4 max_iter=50 eps=1e-06
Iterações: 8 | SSE final: 1347106.704014 | Tempo: 0.4 ms
Coeficiente silhouette médio: 0.669 | Tempo: 102.5 ms

-> [OpenMP] Iniciando testes de escalabilidade...
   Running OMP with 1 threads...
OpenMP habilitado com 1 threads configuradas.
Threads efetivamente utilizadas: 1
K-means 1D (naive)
N=10000 K=4 max_iter=50 eps=1e-06
Iterações: 8 | SSE final: 1347106.704014 | Tempo: 0.4 ms
Coeficiente silhouette médio: 0.669 | Tempo: 104.5 ms
      [VALIDAÇÃO: OK]
   Running OMP with 2 threads...
OpenMP habilitado com 2 threads configuradas.
Threads efetivamente utilizadas: 2
K-means 1D (naive)
N=10000 K=4 max_iter=50 eps=1e-06
Iterações: 8 | SSE final: 1347106.704014 | Tempo: 0.3 ms
Coeficiente silhouette médio: 0.669 | Tempo: 49.9 ms
      [VALIDAÇÃO: OK]
   Running OMP with 4 threads...
OpenMP habilitado com 4 threads configuradas.
Threads efetivamente utilizadas: 4
K-means 1D (naive)
N=10000 K=4 max_iter=50 eps=1e-06
Iterações: 8 | SSE final: 1347106.704014 | Tempo: 0.5 ms
Coeficiente silhouette médio: 0.669 | Tempo: 32.9 ms
      [VALIDAÇÃO: OK]
   Running OMP with 8 threads...
OpenMP habilitado com 8 threads configuradas.
Threads efetivamente utilizadas: 8
K-means 1D (naive)
N=10000 K=4 max_iter=50 eps=1e-06
Iterações: 8 | SSE final: 1347106.704014 | Tempo: 0.6 ms
Coeficiente silhouette médio: 0.669 | Tempo: 17.8 ms
      [VALIDAÇÃO: OK]
   Running OMP with 16 threads...
OpenMP habilitado com 16 threads configuradas.
Threads efetivamente utilizadas: 16
K-means 1D (naive)
N=10000 K=4 max_iter=50 eps=1e-06
Iterações: 8 | SSE final: 1347106.704014 | Tempo: 0.8 ms
Coeficiente silhouette médio: 0.669 | Tempo: 17.6 ms
      [VALIDAÇÃO: OK]
   Running OMP with 32 threads...
OpenMP habilitado com 32 threads configuradas.
Threads efetivamente utilizadas: 32
K-means 1D (naive)
N=10000 K=4 max_iter=50 eps=1e-06
Iterações: 8 | SSE final: 1347106.704014 | Tempo: 1.2 ms
Coeficiente silhouette médio: 0.669 | Tempo: 10.6 ms
      [VALIDAÇÃO: OK]
   Running OMP with 64 threads...
OpenMP habilitado com 64 threads configuradas.
Threads efetivamente utilizadas: 64
K-means 1D (naive)
N=10000 K=4 max_iter=50 eps=1e-06
Iterações: 8 | SSE final: 1347106.704014 | Tempo: 1.9 ms
Coeficiente silhouette médio: 0.669 | Tempo: 7.8 ms
      [VALIDAÇÃO: OK]

-> [MPI] Iniciando testes de escalabilidade...
   Running MPI with 1 processes...
K-means 1D (MPI Distribuído)
N=10000 K=4 P=1 processos
Iterações: 8 | SSE final: 1347106.704014
Tempo Total K-means: 1.2500 ms
Tempo Comunicação (Allreduce): 0.0049 ms (0.4%)
Tempo Silhouette: 104.2399 ms
Coeficiente silhouette médio: 0.669
      [VALIDAÇÃO: OK]
   Running MPI with 2 processes...
K-means 1D (MPI Distribuído)
N=10000 K=4 P=2 processos
Iterações: 8 | SSE final: 1347106.704014
Tempo Total K-means: 0.8006 ms
Tempo Comunicação (Allreduce): 0.1436 ms (17.9%)
Tempo Silhouette: 56.6238 ms
Coeficiente silhouette médio: 0.669
      [VALIDAÇÃO: OK]
   Running MPI with 4 processes...
K-means 1D (MPI Distribuído)
N=10000 K=4 P=4 processos
Iterações: 8 | SSE final: 1347106.704014
Tempo Total K-means: 0.1263 ms
Tempo Comunicação (Allreduce): 0.0514 ms (40.7%)
Tempo Silhouette: 24.1559 ms
Coeficiente silhouette médio: 0.669
      [VALIDAÇÃO: OK]
   Running MPI with 8 processes...
K-means 1D (MPI Distribuído)
N=10000 K=4 P=8 processos
Iterações: 8 | SSE final: 1347106.704014
Tempo Total K-means: 0.5238 ms
Tempo Comunicação (Allreduce): 0.3966 ms (75.7%)
Tempo Silhouette: 24.3006 ms
Coeficiente silhouette médio: 0.669
      [VALIDAÇÃO: OK]
   Running MPI with 16 processes...
K-means 1D (MPI Distribuído)
N=10000 K=4 P=16 processos
Iterações: 8 | SSE final: 1347106.704014
Tempo Total K-means: 2.8228 ms
Tempo Comunicação (Allreduce): 2.7640 ms (97.9%)
Tempo Silhouette: 22.3631 ms
Coeficiente silhouette médio: 0.669
      [VALIDAÇÃO: OK]
   Running MPI with 32 processes...
K-means 1D (MPI Distribuído)
N=10000 K=4 P=32 processos
Iterações: 8 | SSE final: 1347106.704014
Tempo Total K-means: 1.7930 ms
Tempo Comunicação (Allreduce): 1.7605 ms (98.2%)
Tempo Silhouette: 21.7371 ms
Coeficiente silhouette médio: 0.669
      [VALIDAÇÃO: OK]
   Running MPI with 64 processes...
K-means 1D (MPI Distribuído)
N=10000 K=4 P=64 processos
Iterações: 8 | SSE final: 1347106.704014
Tempo Total K-means: 5.2449 ms
Tempo Comunicação (Allreduce): 5.2254 ms (99.6%)
Tempo Silhouette: 23.7096 ms
Coeficiente silhouette médio: 0.669
      [VALIDAÇÃO: OK]

-> [CUDA] Testando configurações de kernel...
   Running CUDA with Block Size = 64...
K-means 1D (CUDA - Opção A)
N=10000 K=4 max_iter=50 eps=1e-06 threadsPerBlock=64
Iterações: 8 | SSE final: 1347106.704014
--- Tempos K-means (ms) ---
  Tempo H2D (cópias C): 0.0 ms
  Tempo Kernel (GPU):   3.8 ms
  Tempo D2H (cópias A): 0.3 ms
  Tempo Total K-means:  4.4 ms
--- Tempos Outros (ms) ---
  Tempo Silhouette (GPU): 10.9 ms
Coeficiente silhouette médio: 0.669190
      [VALIDAÇÃO: OK]
   Running CUDA with Block Size = 128...
K-means 1D (CUDA - Opção A)
N=10000 K=4 max_iter=50 eps=1e-06 threadsPerBlock=128
Iterações: 8 | SSE final: 1347106.704014
--- Tempos K-means (ms) ---
  Tempo H2D (cópias C): 0.2 ms
  Tempo Kernel (GPU):   0.1 ms
  Tempo D2H (cópias A): 0.3 ms
  Tempo Total K-means:  1.0 ms
--- Tempos Outros (ms) ---
  Tempo Silhouette (GPU): 10.1 ms
Coeficiente silhouette médio: 0.669190
      [VALIDAÇÃO: OK]
   Running CUDA with Block Size = 256...
K-means 1D (CUDA - Opção A)
N=10000 K=4 max_iter=50 eps=1e-06 threadsPerBlock=256
Iterações: 8 | SSE final: 1347106.704014
--- Tempos K-means (ms) ---
  Tempo H2D (cópias C): 0.0 ms
  Tempo Kernel (GPU):   0.1 ms
  Tempo D2H (cópias A): 0.3 ms
  Tempo Total K-means:  0.8 ms
--- Tempos Outros (ms) ---
  Tempo Silhouette (GPU): 10.0 ms
Coeficiente silhouette médio: 0.669190
      [VALIDAÇÃO: OK]
   Running CUDA with Block Size = 512...
K-means 1D (CUDA - Opção A)
N=10000 K=4 max_iter=50 eps=1e-06 threadsPerBlock=512
Iterações: 8 | SSE final: 1347106.704014
--- Tempos K-means (ms) ---
  Tempo H2D (cópias C): 0.0 ms
  Tempo Kernel (GPU):   0.1 ms
  Tempo D2H (cópias A): 0.3 ms
  Tempo Total K-means:  0.7 ms
--- Tempos Outros (ms) ---
  Tempo Silhouette (GPU): 10.7 ms
Coeficiente silhouette médio: 0.669190
      [VALIDAÇÃO: OK]
   Running CUDA with Block Size = 1024...
K-means 1D (CUDA - Opção A)
N=10000 K=4 max_iter=50 eps=1e-06 threadsPerBlock=1024
Iterações: 8 | SSE final: 1347106.704014
--- Tempos K-means (ms) ---
  Tempo H2D (cópias C): 0.0 ms
  Tempo Kernel (GPU):   0.1 ms
  Tempo D2H (cópias A): 0.3 ms
  Tempo Total K-means:  0.7 ms
--- Tempos Outros (ms) ---
  Tempo Silhouette (GPU): 10.3 ms
Coeficiente silhouette médio: 0.669190
      [VALIDAÇÃO: OK]

Configuração N=10000 finalizada.
------------------------------------------------------------------
==================================================================
DATASET: N=100000 pontos, K=8 clusters
==================================================================
-> Gerando arquivos de entrada...
-> [SERIAL] Executando Baseline...
K-means 1D (naive)
N=100000 K=8 max_iter=50 eps=1e-06
Iterações: 12 | SSE final: 43979309.268086 | Tempo: 9.6 ms
Coeficiente silhouette médio: 0.600 | Tempo: 28344.2 ms

-> [OpenMP] Iniciando testes de escalabilidade...
   Running OMP with 1 threads...
OpenMP habilitado com 1 threads configuradas.
Threads efetivamente utilizadas: 1
K-means 1D (naive)
N=100000 K=8 max_iter=50 eps=1e-06
Iterações: 12 | SSE final: 43979309.268086 | Tempo: 5.2 ms
Coeficiente silhouette médio: 0.600 | Tempo: 28155.7 ms
      [VALIDAÇÃO: OK]
   Running OMP with 2 threads...
OpenMP habilitado com 2 threads configuradas.
Threads efetivamente utilizadas: 2
K-means 1D (naive)
N=100000 K=8 max_iter=50 eps=1e-06
Iterações: 12 | SSE final: 43979309.268086 | Tempo: 4.3 ms
Coeficiente silhouette médio: 0.600 | Tempo: 14715.9 ms
      [VALIDAÇÃO: OK]
   Running OMP with 4 threads...
OpenMP habilitado com 4 threads configuradas.
Threads efetivamente utilizadas: 4
K-means 1D (naive)
N=100000 K=8 max_iter=50 eps=1e-06
Iterações: 12 | SSE final: 43979309.268087 | Tempo: 2.8 ms
Coeficiente silhouette médio: 0.600 | Tempo: 7328.0 ms
      [VALIDAÇÃO: OK]
   Running OMP with 8 threads...
OpenMP habilitado com 8 threads configuradas.
Threads efetivamente utilizadas: 8
K-means 1D (naive)
N=100000 K=8 max_iter=50 eps=1e-06
Iterações: 12 | SSE final: 43979309.268087 | Tempo: 1.7 ms
Coeficiente silhouette médio: 0.600 | Tempo: 3657.9 ms
      [VALIDAÇÃO: OK]
   Running OMP with 16 threads...
OpenMP habilitado com 16 threads configuradas.
Threads efetivamente utilizadas: 16
K-means 1D (naive)
N=100000 K=8 max_iter=50 eps=1e-06
Iterações: 12 | SSE final: 43979309.268086 | Tempo: 1.8 ms
Coeficiente silhouette médio: 0.600 | Tempo: 2841.5 ms
      [VALIDAÇÃO: OK]
   Running OMP with 32 threads...
OpenMP habilitado com 32 threads configuradas.
Threads efetivamente utilizadas: 32
K-means 1D (naive)
N=100000 K=8 max_iter=50 eps=1e-06
Iterações: 12 | SSE final: 43979309.268087 | Tempo: 2.3 ms
Coeficiente silhouette médio: 0.600 | Tempo: 1813.1 ms
      [VALIDAÇÃO: OK]
   Running OMP with 64 threads...
OpenMP habilitado com 64 threads configuradas.
Threads efetivamente utilizadas: 64
K-means 1D (naive)
N=100000 K=8 max_iter=50 eps=1e-06
Iterações: 12 | SSE final: 43979309.268087 | Tempo: 3.0 ms
Coeficiente silhouette médio: 0.600 | Tempo: 1756.6 ms
      [VALIDAÇÃO: OK]

-> [MPI] Iniciando testes de escalabilidade...
   Running MPI with 1 processes...
K-means 1D (MPI Distribuído)
N=100000 K=8 P=1 processos
Iterações: 12 | SSE final: 43979309.268086
Tempo Total K-means: 5.4136 ms
Tempo Comunicação (Allreduce): 0.0029 ms (0.1%)
Tempo Silhouette: 29399.0661 ms
Coeficiente silhouette médio: 0.600
      [VALIDAÇÃO: OK]
   Running MPI with 2 processes...
K-means 1D (MPI Distribuído)
N=100000 K=8 P=2 processos
Iterações: 12 | SSE final: 43979309.268086
Tempo Total K-means: 2.8393 ms
Tempo Comunicação (Allreduce): 0.2605 ms (9.2%)
Tempo Silhouette: 14548.1132 ms
Coeficiente silhouette médio: 0.600
      [VALIDAÇÃO: OK]
   Running MPI with 4 processes...
K-means 1D (MPI Distribuído)
N=100000 K=8 P=4 processos
Iterações: 12 | SSE final: 43979309.268087
Tempo Total K-means: 1.4868 ms
Tempo Comunicação (Allreduce): 0.1632 ms (11.0%)
Tempo Silhouette: 7253.6635 ms
Coeficiente silhouette médio: 0.600
      [VALIDAÇÃO: OK]
   Running MPI with 8 processes...
K-means 1D (MPI Distribuído)
N=100000 K=8 P=8 processos
Iterações: 12 | SSE final: 43979309.268087
Tempo Total K-means: 1.1161 ms
Tempo Comunicação (Allreduce): 0.3539 ms (31.7%)
Tempo Silhouette: 3690.6039 ms
Coeficiente silhouette médio: 0.600
      [VALIDAÇÃO: OK]
   Running MPI with 16 processes...
K-means 1D (MPI Distribuído)
N=100000 K=8 P=16 processos
Iterações: 12 | SSE final: 43979309.268086
Tempo Total K-means: 0.7960 ms
Tempo Comunicação (Allreduce): 0.2130 ms (26.8%)
Tempo Silhouette: 3374.4687 ms
Coeficiente silhouette médio: 0.600
      [VALIDAÇÃO: OK]
   Running MPI with 32 processes...
K-means 1D (MPI Distribuído)
N=100000 K=8 P=32 processos
Iterações: 12 | SSE final: 43979309.268087
Tempo Total K-means: 1.2909 ms
Tempo Comunicação (Allreduce): 0.9972 ms (77.2%)
Tempo Silhouette: 2321.7883 ms
Coeficiente silhouette médio: 0.600
      [VALIDAÇÃO: OK]
   Running MPI with 64 processes...
K-means 1D (MPI Distribuído)
N=100000 K=8 P=64 processos
Iterações: 12 | SSE final: 43979309.268087
Tempo Total K-means: 2.9382 ms
Tempo Comunicação (Allreduce): 2.7866 ms (94.8%)
Tempo Silhouette: 3151.4340 ms
Coeficiente silhouette médio: 0.600
      [VALIDAÇÃO: OK]

-> [CUDA] Testando configurações de kernel...
   Running CUDA with Block Size = 64...
K-means 1D (CUDA - Opção A)
N=100000 K=8 max_iter=50 eps=1e-06 threadsPerBlock=64
Iterações: 12 | SSE final: 43979309.268086
--- Tempos K-means (ms) ---
  Tempo H2D (cópias C): 0.1 ms
  Tempo Kernel (GPU):   0.4 ms
  Tempo D2H (cópias A): 2.4 ms
  Tempo Total K-means:  5.0 ms
--- Tempos Outros (ms) ---
  Tempo Silhouette (GPU): 1818.9 ms
Coeficiente silhouette médio: 0.600280
      [VALIDAÇÃO: OK]
   Running CUDA with Block Size = 128...
K-means 1D (CUDA - Opção A)
N=100000 K=8 max_iter=50 eps=1e-06 threadsPerBlock=128
Iterações: 12 | SSE final: 43979309.268086
--- Tempos K-means (ms) ---
  Tempo H2D (cópias C): 0.2 ms
  Tempo Kernel (GPU):   0.5 ms
  Tempo D2H (cópias A): 2.4 ms
  Tempo Total K-means:  5.4 ms
--- Tempos Outros (ms) ---
  Tempo Silhouette (GPU): 1824.1 ms
Coeficiente silhouette médio: 0.600280
      [VALIDAÇÃO: OK]
   Running CUDA with Block Size = 256...
K-means 1D (CUDA - Opção A)
N=100000 K=8 max_iter=50 eps=1e-06 threadsPerBlock=256
Iterações: 12 | SSE final: 43979309.268086
--- Tempos K-means (ms) ---
  Tempo H2D (cópias C): 0.2 ms
  Tempo Kernel (GPU):   0.4 ms
  Tempo D2H (cópias A): 2.4 ms
  Tempo Total K-means:  5.8 ms
--- Tempos Outros (ms) ---
  Tempo Silhouette (GPU): 1810.2 ms
Coeficiente silhouette médio: 0.600280
      [VALIDAÇÃO: OK]
   Running CUDA with Block Size = 512...
K-means 1D (CUDA - Opção A)
N=100000 K=8 max_iter=50 eps=1e-06 threadsPerBlock=512
Iterações: 12 | SSE final: 43979309.268086
--- Tempos K-means (ms) ---
  Tempo H2D (cópias C): 0.1 ms
  Tempo Kernel (GPU):   0.4 ms
  Tempo D2H (cópias A): 2.4 ms
  Tempo Total K-means:  5.2 ms
--- Tempos Outros (ms) ---
  Tempo Silhouette (GPU): 1815.1 ms
Coeficiente silhouette médio: 0.600280
      [VALIDAÇÃO: OK]
   Running CUDA with Block Size = 1024...
K-means 1D (CUDA - Opção A)
N=100000 K=8 max_iter=50 eps=1e-06 threadsPerBlock=1024
Iterações: 12 | SSE final: 43979309.268086
--- Tempos K-means (ms) ---
  Tempo H2D (cópias C): 0.1 ms
  Tempo Kernel (GPU):   0.5 ms
  Tempo D2H (cópias A): 2.4 ms
  Tempo Total K-means:  5.1 ms
--- Tempos Outros (ms) ---
  Tempo Silhouette (GPU): 1809.1 ms
Coeficiente silhouette médio: 0.600280
      [VALIDAÇÃO: OK]

Configuração N=100000 finalizada.
------------------------------------------------------------------
==================================================================
DATASET: N=1000000 pontos, K=16 clusters
==================================================================
-> Gerando arquivos de entrada...
-> [SERIAL] Executando Baseline...
K-means 1D (naive)
N=1000000 K=16 max_iter=50 eps=1e-06
Iterações: 40 | SSE final: 1238641281.008392 | Tempo: 288.0 ms
Coeficiente silhouette médio: 0.586 | Tempo: 6133119.0 ms

-> [OpenMP] Iniciando testes de escalabilidade...
   Running OMP with 1 threads...
OpenMP habilitado com 1 threads configuradas.
Threads efetivamente utilizadas: 1
K-means 1D (naive)
N=1000000 K=16 max_iter=50 eps=1e-06
Iterações: 40 | SSE final: 1238641281.008392 | Tempo: 289.0 ms
Coeficiente silhouette médio: 0.586 | Tempo: 6124496.0 ms
      [VALIDAÇÃO: OK]
   Running OMP with 2 threads...
OpenMP habilitado com 2 threads configuradas.
Threads efetivamente utilizadas: 2
K-means 1D (naive)
N=1000000 K=16 max_iter=50 eps=1e-06
Iterações: 40 | SSE final: 1238641281.008402 | Tempo: 156.6 ms
Coeficiente silhouette médio: 0.586 | Tempo: 3167212.3 ms
      [VALIDAÇÃO: OK]
   Running OMP with 4 threads...
OpenMP habilitado com 4 threads configuradas.
Threads efetivamente utilizadas: 4
K-means 1D (naive)
N=1000000 K=16 max_iter=50 eps=1e-06
Iterações: 40 | SSE final: 1238641281.008427 | Tempo: 79.8 ms
Coeficiente silhouette médio: 0.586 | Tempo: 1581141.3 ms
      [VALIDAÇÃO: OK]
   Running OMP with 8 threads...
OpenMP habilitado com 8 threads configuradas.
Threads efetivamente utilizadas: 8
K-means 1D (naive)
N=1000000 K=16 max_iter=50 eps=1e-06
Iterações: 40 | SSE final: 1238641281.008409 | Tempo: 55.1 ms
Coeficiente silhouette médio: 0.586 | Tempo: 838512.7 ms
      [VALIDAÇÃO: OK]
   Running OMP with 16 threads...
OpenMP habilitado com 16 threads configuradas.
Threads efetivamente utilizadas: 16
K-means 1D (naive)
N=1000000 K=16 max_iter=50 eps=1e-06
Iterações: 40 | SSE final: 1238641281.008405 | Tempo: 39.7 ms
Coeficiente silhouette médio: 0.586 | Tempo: 655215.4 ms
      [VALIDAÇÃO: OK]
   Running OMP with 32 threads...
OpenMP habilitado com 32 threads configuradas.
Threads efetivamente utilizadas: 32
K-means 1D (naive)
N=1000000 K=16 max_iter=50 eps=1e-06
Iterações: 40 | SSE final: 1238641281.008404 | Tempo: 30.3 ms
Coeficiente silhouette médio: 0.586 | Tempo: 421471.9 ms
      [VALIDAÇÃO: OK]
   Running OMP with 64 threads...
OpenMP habilitado com 64 threads configuradas.
Threads efetivamente utilizadas: 64
K-means 1D (naive)
N=1000000 K=16 max_iter=50 eps=1e-06
Iterações: 40 | SSE final: 1238641281.008404 | Tempo: 35.9 ms
Coeficiente silhouette médio: 0.586 | Tempo: 416715.5 ms
      [VALIDAÇÃO: OK]

-> [MPI] Iniciando testes de escalabilidade...
   Running MPI with 1 processes...
K-means 1D (MPI Distribuído)
N=1000000 K=16 P=1 processos
Iterações: 40 | SSE final: 1238641281.008392
Tempo Total K-means: 292.5193 ms
Tempo Comunicação (Allreduce): 0.0203 ms (0.0%)
Tempo Silhouette: 6311625.6716 ms
Coeficiente silhouette médio: 0.586
      [VALIDAÇÃO: OK]
   Running MPI with 2 processes...
K-means 1D (MPI Distribuído)
N=1000000 K=16 P=2 processos
Iterações: 40 | SSE final: 1238641281.008402
Tempo Total K-means: 153.2228 ms
Tempo Comunicação (Allreduce): 10.6689 ms (7.0%)
Tempo Silhouette: 3179670.8506 ms
Coeficiente silhouette médio: 0.586
      [VALIDAÇÃO: OK]
   Running MPI with 4 processes...
K-means 1D (MPI Distribuído)
N=1000000 K=16 P=4 processos
Iterações: 40 | SSE final: 1238641281.008427
Tempo Total K-means: 77.8640 ms
Tempo Comunicação (Allreduce): 7.3759 ms (9.5%)
Tempo Silhouette: 1585996.2256 ms
Coeficiente silhouette médio: 0.586
      [VALIDAÇÃO: OK]
   Running MPI with 8 processes...
K-means 1D (MPI Distribuído)
N=1000000 K=16 P=8 processos
Iterações: 40 | SSE final: 1238641281.008409
Tempo Total K-means: 44.9852 ms
Tempo Comunicação (Allreduce): 8.5382 ms (19.0%)
Tempo Silhouette: 882934.1799 ms
Coeficiente silhouette médio: 0.586
      [VALIDAÇÃO: OK]
   Running MPI with 16 processes...
K-means 1D (MPI Distribuído)
N=1000000 K=16 P=16 processos
Iterações: 40 | SSE final: 1238641281.008405
Tempo Total K-means: 31.3533 ms
Tempo Comunicação (Allreduce): 2.9107 ms (9.3%)
Tempo Silhouette: 959781.7340 ms
Coeficiente silhouette médio: 0.586
      [VALIDAÇÃO: OK]
   Running MPI with 32 processes...
K-means 1D (MPI Distribuído)
N=1000000 K=16 P=32 processos
Iterações: 40 | SSE final: 1238641281.008404
Tempo Total K-means: 88.7795 ms
Tempo Comunicação (Allreduce): 70.4782 ms (79.4%)
Tempo Silhouette: 1110271.2150 ms
Coeficiente silhouette médio: 0.586
      [VALIDAÇÃO: OK]
   Running MPI with 64 processes...
K-means 1D (MPI Distribuído)
N=1000000 K=16 P=64 processos
Iterações: 40 | SSE final: 1238641281.008404
Tempo Total K-means: 54.7587 ms
Tempo Comunicação (Allreduce): 46.2320 ms (84.4%)
Tempo Silhouette: 1124811.0685 ms
Coeficiente silhouette médio: 0.586
      [VALIDAÇÃO: OK]

-> [CUDA] Testando configurações de kernel...
   Running CUDA with Block Size = 64...
K-means 1D (CUDA - Opção A)
N=1000000 K=16 max_iter=50 eps=1e-06 threadsPerBlock=64
Iterações: 40 | SSE final: 1238641281.008392
--- Tempos K-means (ms) ---
  Tempo H2D (cópias C): 0.9 ms
  Tempo Kernel (GPU):   19.5 ms
  Tempo D2H (cópias A): 44.3 ms
  Tempo Total K-means:  125.7 ms
--- Tempos Outros (ms) ---
  Tempo Silhouette (GPU): 421333.9 ms
Coeficiente silhouette médio: 0.586
      [VALIDAÇÃO: OK]
   Running CUDA with Block Size = 128...
K-means 1D (CUDA - Opção A)
N=1000000 K=16 max_iter=50 eps=1e-06 threadsPerBlock=128
Iterações: 40 | SSE final: 1238641281.008392
--- Tempos K-means (ms) ---
  Tempo H2D (cópias C): 0.9 ms
  Tempo Kernel (GPU):   19.5 ms
  Tempo D2H (cópias A): 44.1 ms
  Tempo Total K-means:  125.3 ms
--- Tempos Outros (ms) ---
  Tempo Silhouette (GPU): 421992.9 ms
Coeficiente silhouette médio: 0.586
      [VALIDAÇÃO: OK]
   Running CUDA with Block Size = 256...
K-means 1D (CUDA - Opção A)
N=1000000 K=16 max_iter=50 eps=1e-06 threadsPerBlock=256
Iterações: 40 | SSE final: 1238641281.008392
--- Tempos K-means (ms) ---
  Tempo H2D (cópias C): 0.7 ms
  Tempo Kernel (GPU):   19.7 ms
  Tempo D2H (cópias A): 44.2 ms
  Tempo Total K-means:  125.5 ms
--- Tempos Outros (ms) ---
  Tempo Silhouette (GPU): 441471.9 ms
Coeficiente silhouette médio: 0.586
      [VALIDAÇÃO: OK]
   Running CUDA with Block Size = 512...
K-means 1D (CUDA - Opção A)
N=1000000 K=16 max_iter=50 eps=1e-06 threadsPerBlock=512
Iterações: 40 | SSE final: 1238641281.008392
--- Tempos K-means (ms) ---
  Tempo H2D (cópias C): 0.8 ms
  Tempo Kernel (GPU):   19.6 ms
  Tempo D2H (cópias A): 44.3 ms
  Tempo Total K-means:  125.8 ms
--- Tempos Outros (ms) ---
  Tempo Silhouette (GPU): 424471.9 ms
Coeficiente silhouette médio: 0.586
      [VALIDAÇÃO: OK]
   Running CUDA with Block Size = 1024...
K-means 1D (CUDA - Opção A)
N=1000000 K=16 max_iter=50 eps=1e-06 threadsPerBlock=1024
Iterações: 40 | SSE final: 1238641281.008392
--- Tempos K-means (ms) ---
  Tempo H2D (cópias C): 0.8 ms
  Tempo Kernel (GPU):   20.2 ms
  Tempo D2H (cópias A): 44.1 ms
  Tempo Total K-means:  125.9 ms
--- Tempos Outros (ms) ---
  Tempo Silhouette (GPU): 421871.9 ms
Coeficiente silhouette médio: 0.586
      [VALIDAÇÃO: OK]

Configuração N=1000000 finalizada.
------------------------------------------------------------------
Experimentos concluídos. Verifique results/final_results_overnight_20251211_185730.txt
"""


def parse_report(text):
    data = []
    current_n = 0
    current_tech = None
    current_config = 0
    
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Regex Parsers
        dataset_match = re.search(r"DATASET: N=(\d+)", line)
        if dataset_match:
            current_n = int(dataset_match.group(1))
            continue
            
        serial_match = re.search(r"-> \[SERIAL\]", line)
        if serial_match:
            current_tech = "Serial"
            current_config = 1
            continue
            
        omp_match = re.search(r"Running OMP with (\d+) threads", line)
        if omp_match:
            current_tech = "OpenMP"
            current_config = int(omp_match.group(1))
            continue
            
        mpi_match = re.search(r"Running MPI with (\d+) processes", line)
        if mpi_match:
            current_tech = "MPI"
            current_config = int(mpi_match.group(1))
            continue
            
        cuda_match = re.search(r"Running CUDA with Block Size = (\d+)", line)
        if cuda_match:
            current_tech = "CUDA"
            current_config = int(cuda_match.group(1))
            continue
            
        # Extração de Métricas
        # Padrão para SSE (comum a todos)
        sse_match = re.search(r"SSE final: ([\d.]+)", line)
        
        # Padrão para Tempo (Serial/OMP ficam na mesma linha ou próxima ao SSE)
        time_simple_match = re.search(r"Tempo: ([\d.]+) ms", line)
        
        # Padrão para Tempo MPI/CUDA (mais explícito)
        time_full_match = re.search(r"Tempo Total K-means:\s+([\d.]+) ms", line)
        
        # Padrão para Silhouette
        sil_match = re.search(r"Coeficiente silhouette médio: ([\d.]+)", line)
        
        # Lógica de inserção
        # Nota: O Serial tem "Tempo: X" na linha de SSE e "Tempo: Y" na linha Silhouette.
        # Queremos o Tempo do K-means (linha SSE)
        
        if sse_match:
            sse_val = float(sse_match.group(1))
            
            # Se for Serial ou OpenMP, o tempo costuma estar nesta mesma linha
            if current_tech in ["Serial", "OpenMP"] and time_simple_match:
                time_val = float(time_simple_match.group(1))
                # Silhouette geralmente vem na próxima linha, inicializamos como None
                data.append({
                    "N": current_n, "Tech": current_tech, "Config": current_config, 
                    "Time_ms": time_val, "SSE": sse_val, "Silhouette": None
                })
            # MPI e CUDA capturamos o SSE, mas o tempo vem depois
            elif current_tech in ["MPI", "CUDA"]:
                 data.append({
                    "N": current_n, "Tech": current_tech, "Config": current_config, 
                    "Time_ms": None, "SSE": sse_val, "Silhouette": None
                })

        # Preenchendo dados faltantes (Tempo MPI/CUDA ou Silhouette)
        if time_full_match and current_tech in ["MPI", "CUDA"]:
            # Atualiza o último registro adicionado
            if data and data[-1]["Tech"] == current_tech:
                data[-1]["Time_ms"] = float(time_full_match.group(1))
                
        if sil_match:
            if data:
                data[-1]["Silhouette"] = float(sil_match.group(1))

    return pd.DataFrame(data)

def process_data(df):
    # Calcular Speedup
    # Speedup = Tempo_Serial / Tempo_Paralelo (para o mesmo N)
    df['Speedup'] = 0.0
    
    # Isolar tempos seriais
    serial_times = df[df['Tech'] == 'Serial'].set_index('N')['Time_ms'].to_dict()
    
    for index, row in df.iterrows():
        n = row['N']
        if n in serial_times and row['Time_ms'] > 0:
            df.at[index, 'Speedup'] = serial_times[n] / row['Time_ms']
            
    return df

def generate_plots(df):
    # Configurações visuais
    sns.set_theme(style="whitegrid")
    
    # Filtro para o maior Dataset (onde a escalabilidade importa mais)
    max_n = df['N'].max()
    df_max = df[df['N'] == max_n].copy()
    
    # Criar figura com subplots (2 linhas, 2 colunas)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Análise de Desempenho PCD (Dataset N={max_n})', fontsize=16)

    # ---------------------------------------------------------
    # Gráfico 1: Speedup vs Recursos (OpenMP e MPI)
    # ---------------------------------------------------------
    # CUDA não entra aqui pois BlockSize não é "Core Count" linear
    subset_speedup = df_max[df_max['Tech'].isin(['OpenMP', 'MPI'])]
    
    sns.lineplot(data=subset_speedup, x='Config', y='Speedup', hue='Tech', marker='o', style='Tech', ax=axes[0, 0], linewidth=2)
    
    # Linha Ideal
    min_config = subset_speedup['Config'].min()
    max_config = subset_speedup['Config'].max()
    axes[0, 0].plot([min_config, max_config], [min_config, max_config], 'k--', alpha=0.5, label='Ideal Linear')
    
    axes[0, 0].set_title('Speedup vs Recursos (Lei de Amdahl/Gustafson)')
    axes[0, 0].set_xlabel('Número de Threads/Processos')
    axes[0, 0].set_ylabel('Speedup (T_serial / T_paralelo)')
    axes[0, 0].set_xscale('log', base=2)
    axes[0, 0].set_yscale('log', base=2)
    axes[0, 0].legend()

    # ---------------------------------------------------------
    # Gráfico 2: Tempo de Execução vs Configuração (Todas as Techs)
    # ---------------------------------------------------------
    # Aqui comparamos Serial, OpenMP, MPI e CUDA
    # Serial será uma linha horizontal constante
    serial_time = df_max[df_max['Tech'] == 'Serial']['Time_ms'].values[0]
    
    sns.lineplot(data=df_max[df_max['Tech'] != 'Serial'], x='Config', y='Time_ms', hue='Tech', marker='o', ax=axes[0, 1])
    axes[0, 1].axhline(y=serial_time, color='r', linestyle='--', label=f'Serial ({serial_time:.1f}ms)')
    
    axes[0, 1].set_title('Tempo de Execução (Escala Log)')
    axes[0, 1].set_xlabel('Configuração (Threads/Procs/Blocks)')
    axes[0, 1].set_ylabel('Tempo (ms)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()

    # ---------------------------------------------------------
    # Gráfico 3: Corretude (SSE e Silhouette)
    # ---------------------------------------------------------
    # Vamos usar um Scatter plot para mostrar que os valores não variam
    # Normalizar SSE para visualização (dividir por 1e9 para N=1M)
    df_max['SSE_Scaled'] = df_max['SSE'] / 1e9
    
    # Criar um eixo gêmeo para mostrar Silhouette e SSE juntos
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    
    sns.scatterplot(data=df_max, x='Tech', y='SSE_Scaled', color='blue', s=100, label='SSE (x10^9)', ax=ax3, legend=False)
    sns.scatterplot(data=df_max, x='Tech', y='Silhouette', color='orange', marker='X', s=100, label='Silhouette', ax=ax3_twin, legend=False)
    
    ax3.set_title('Validação de Corretude (Estabilidade dos Resultados)')
    ax3.set_ylabel('SSE (Soma dos Quadrados)')
    ax3_twin.set_ylabel('Coeficiente Silhouette')
    
    # Ajustar legendas manuais
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper right')

    # ---------------------------------------------------------
    # Gráfico 4: Melhor Tempo por Tecnologia (Bar Chart)
    # ---------------------------------------------------------
    # Pegar o menor tempo de cada tecnologia
    best_times = df_max.loc[df_max.groupby('Tech')['Time_ms'].idxmin()]
    
    barplot = sns.barplot(data=best_times, x='Tech', y='Time_ms', hue='Tech', palette='viridis', ax=axes[1, 1], legend=False)
    axes[1, 1].set_title(f'Melhor Tempo Absoluto (N={max_n})')
    axes[1, 1].set_ylabel('Tempo (ms)')
    
    # Adicionar labels nas barras
    for container in axes[1, 1].containers:
        axes[1, 1].bar_label(container, fmt='%.1f ms')

    plt.tight_layout()
    plt.savefig('pcd_resultados.png')
    plt.show()

# --- Execução Principal ---
df = parse_report(raw_data)
df = process_data(df)
df.to_csv("pcd_dados_processados.csv", index=False)
print("CSV gerado: pcd_dados_processados.csv")
generate_plots(df)