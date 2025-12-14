import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cole o conteúdo COMPLETO do seu arquivo txt aqui (mesmo do plot_results_overnight.py)
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

def parse_cuda_timings(text):
    """
    Extrai os tempos H2D, Kernel e D2H das execuções CUDA
    """
    data = []
    current_n = None
    current_block = None
    
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Detecta o dataset atual
        dataset_match = re.search(r"DATASET: N=(\d+)", line)
        if dataset_match:
            current_n = int(dataset_match.group(1))
            continue
        
        # Detecta o block size
        block_match = re.search(r"Running CUDA with Block Size = (\d+)", line)
        if block_match:
            current_block = int(block_match.group(1))
            continue
        
        # Captura os tempos (H2D, Kernel, D2H)
        h2d_match = re.search(r"Tempo H2D \(cópias C\):\s+([\d.]+) ms", line)
        kernel_match = re.search(r"Tempo Kernel \(GPU\):\s+([\d.]+) ms", line)
        d2h_match = re.search(r"Tempo D2H \(cópias A\):\s+([\d.]+) ms", line)
        
        if h2d_match and current_n and current_block:
            h2d_time = float(h2d_match.group(1))
            
            # Próximas linhas devem conter Kernel e D2H
            kernel_time = None
            d2h_time = None
            
            for j in range(i+1, min(i+5, len(lines))):
                next_line = lines[j].strip()
                if not kernel_time:
                    km = re.search(r"Tempo Kernel \(GPU\):\s+([\d.]+) ms", next_line)
                    if km:
                        kernel_time = float(km.group(1))
                if not d2h_time:
                    dm = re.search(r"Tempo D2H \(cópias A\):\s+([\d.]+) ms", next_line)
                    if dm:
                        d2h_time = float(dm.group(1))
                        
            if kernel_time is not None and d2h_time is not None:
                data.append({
                    'N': current_n,
                    'BlockSize': current_block,
                    'H2D': h2d_time,
                    'Kernel': kernel_time,
                    'D2H': d2h_time
                })
    
    return pd.DataFrame(data)

def plot_cuda_breakdown_with_std(df):
    """
    Gera 3 subplots (um por dataset) mostrando H2D, Kernel e D2H
    com barras de erro representando o desvio padrão entre diferentes block sizes
    """
    sns.set_theme(style="whitegrid")
    
    datasets = sorted(df['N'].unique())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('CUDA K-means: Breakdown de Tempo por Componente (H2D, Kernel, D2H)', fontsize=16)
    
    for idx, n in enumerate(datasets):
        df_n = df[df['N'] == n]
        
        # Calcula média e desvio padrão para cada componente
        means = {
            'H2D': df_n['H2D'].mean(),
            'Kernel': df_n['Kernel'].mean(),
            'D2H': df_n['D2H'].mean()
        }
        
        stds = {
            'H2D': df_n['H2D'].std(),
            'Kernel': df_n['Kernel'].std(),
            'D2H': df_n['D2H'].std()
        }
        
        # Dados para o barplot
        components = list(means.keys())
        mean_values = list(means.values())
        std_values = list(stds.values())
        
        # Cria o barplot com barras de erro
        ax = axes[idx]
        bars = ax.bar(components, mean_values, yerr=std_values, 
                     capsize=5, alpha=0.8, color=['#2c7bb6', '#abd9e9', '#fdae61'])
        
        # Adiciona labels nos topos das barras
        for bar, mean_val, std_val in zip(bars, mean_values, std_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std_val,
                   f'{mean_val:.2f}±{std_val:.2f}',
                   ha='center', va='bottom', fontsize=9)
        
        # Formatação
        dbk_label = int(np.log10(n) - 3)
        ax.set_title(f'DBK{dbk_label} (N={n:,} pontos)')
        ax.set_ylabel('Tempo (ms)')
        ax.set_xlabel('Componente')
        ax.grid(axis='y', alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('cuda_breakdown_with_std.png', dpi=300)
    print("Gráfico salvo como 'cuda_breakdown_with_std.png'")
    plt.show()

# --- Execução Principal ---
if __name__ == "__main__":
    df_cuda = parse_cuda_timings(raw_data)
    
    if df_cuda.empty:
        print("ERRO: Nenhum dado CUDA foi encontrado no log.")
    else:
        print("Dados CUDA extraídos:")
        print(df_cuda)
        print("\nEstatísticas por Dataset:")
        print(df_cuda.groupby('N')[['H2D', 'Kernel', 'D2H']].agg(['mean', 'std']))
        
        plot_cuda_breakdown_with_std(df_cuda)