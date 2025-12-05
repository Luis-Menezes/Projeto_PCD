import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np

# ==========================================
# 1. ENTRADA DE DADOS (SEU LOG ATUALIZADO)
# ==========================================
mpi_log = """
=== K-MEANS MPI COMPREHENSIVE TEST RESULTS ===
Data e hora: Thu Dec  4 02:14:53 PM -03 2025
========================================
DATASET: 10000 pontos, 4 clusters
========================================
--- MPI com 1 processos ---
K-means 1D (MPI Distribuído)
N=10000 K=4 P=1 processos
Iterações: 3 | SSE final: 84195.049605
Tempo Total K-means: 0.0004 s
Tempo Comunicação (Allreduce): 0.0000 s (0.9%)
Tempo Silhouette: 0.1045 s
--- MPI com 2 processos ---
K-means 1D (MPI Distribuído)
N=10000 K=4 P=2 processos
Iterações: 3 | SSE final: 84195.049605
Tempo Total K-means: 0.0004 s
Tempo Comunicação (Allreduce): 0.0001 s (24.1%)
Tempo Silhouette: 0.0556 s
--- MPI com 4 processos ---
K-means 1D (MPI Distribuído)
N=10000 K=4 P=4 processos
Iterações: 3 | SSE final: 84195.049605
Tempo Total K-means: 0.0001 s
Tempo Comunicação (Allreduce): 0.0000 s (28.8%)
Tempo Silhouette: 0.0269 s
--- MPI com 8 processos ---
K-means 1D (MPI Distribuído)
N=10000 K=4 P=8 processos
Iterações: 3 | SSE final: 84195.049605
Tempo Total K-means: 0.0002 s
Tempo Comunicação (Allreduce): 0.0002 s (82.6%)
Tempo Silhouette: 0.0297 s
--- MPI com 16 processos ---
K-means 1D (MPI Distribuído)
N=10000 K=4 P=16 processos
Iterações: 3 | SSE final: 84195.049605
Tempo Total K-means: 0.0003 s
Tempo Comunicação (Allreduce): 0.0003 s (90.3%)
Tempo Silhouette: 0.0179 s
--- MPI com 32 processos ---
K-means 1D (MPI Distribuído)
N=10000 K=4 P=32 processos
Iterações: 3 | SSE final: 84195.049605
Tempo Total K-means: 0.0005 s
Tempo Comunicação (Allreduce): 0.0005 s (96.3%)
Tempo Silhouette: 0.0213 s
========================================
DATASET: 100000 pontos, 8 clusters
========================================
--- MPI com 1 processos ---
K-means 1D (MPI Distribuído)
N=100000 K=8 P=1 processos
Iterações: 3 | SSE final: 834307.525443
Tempo Total K-means: 0.0014 s
Tempo Comunicação (Allreduce): 0.0000 s (0.1%)
Tempo Silhouette: 16.8051 s
--- MPI com 2 processos ---
K-means 1D (MPI Distribuído)
N=100000 K=8 P=2 processos
Iterações: 3 | SSE final: 834307.525443
Tempo Total K-means: 0.0007 s
Tempo Comunicação (Allreduce): 0.0000 s (1.5%)
Tempo Silhouette: 8.3425 s
--- MPI com 4 processos ---
K-means 1D (MPI Distribuído)
N=100000 K=8 P=4 processos
Iterações: 3 | SSE final: 834307.525443
Tempo Total K-means: 0.0004 s
Tempo Comunicação (Allreduce): 0.0000 s (3.6%)
Tempo Silhouette: 4.1911 s
--- MPI com 8 processos ---
K-means 1D (MPI Distribuído)
N=100000 K=8 P=8 processos
Iterações: 3 | SSE final: 834307.525443
Tempo Total K-means: 0.0003 s
Tempo Comunicação (Allreduce): 0.0001 s (33.9%)
Tempo Silhouette: 2.1245 s
--- MPI com 16 processos ---
K-means 1D (MPI Distribuído)
N=100000 K=8 P=16 processos
Iterações: 3 | SSE final: 834307.525443
Tempo Total K-means: 0.0002 s
Tempo Comunicação (Allreduce): 0.0001 s (28.8%)
Tempo Silhouette: 2.6050 s
--- MPI com 32 processos ---
K-means 1D (MPI Distribuído)
N=100000 K=8 P=32 processos
Iterações: 3 | SSE final: 834307.525443
Tempo Total K-means: 0.0006 s
Tempo Comunicação (Allreduce): 0.0004 s (72.3%)
Tempo Silhouette: 1.6821 s
========================================
DATASET: 1000000 pontos, 16 clusters
========================================
--- MPI com 1 processos ---
K-means 1D (MPI Distribuído)
N=1000000 K=16 P=1 processos
Iterações: 3 | SSE final: 8332982.024638
Tempo Total K-means: 0.0227 s
Tempo Comunicação (Allreduce): 0.0000 s (0.0%)
Tempo Silhouette: 3132.3590 s
--- MPI com 2 processos ---
K-means 1D (MPI Distribuído)
N=1000000 K=16 P=2 processos
Iterações: 3 | SSE final: 8332982.024638
Tempo Total K-means: 0.0116 s
Tempo Comunicação (Allreduce): 0.0001 s (1.0%)
Tempo Silhouette: 1601.0024 s
--- MPI com 4 processos ---
K-means 1D (MPI Distribuído)
N=1000000 K=16 P=4 processos
Iterações: 3 | SSE final: 8332982.024638
Tempo Total K-means: 0.0057 s
Tempo Comunicação (Allreduce): 0.0001 s (2.4%)
Tempo Silhouette: 811.3318 s
--- MPI com 8 processos ---
K-means 1D (MPI Distribuído)
N=1000000 K=16 P=8 processos
Iterações: 3 | SSE final: 8332982.024638
Tempo Total K-means: 0.0034 s
Tempo Comunicação (Allreduce): 0.0004 s (12.4%)
Tempo Silhouette: 453.8107 s
--- MPI com 16 processos ---
K-means 1D (MPI Distribuído)
N=1000000 K=16 P=16 processos
Iterações: 3 | SSE final: 8332982.024638
Tempo Total K-means: 0.0024 s
Tempo Comunicação (Allreduce): 0.0010 s (39.8%)
Tempo Silhouette: 687.0907 s
--- MPI com 32 processos ---
K-means 1D (MPI Distribuído)
N=1000000 K=16 P=32 processos
Iterações: 3 | SSE final: 8332982.024638
Tempo Total K-means: 0.0026 s
Tempo Comunicação (Allreduce): 0.0014 s (54.0%)
Tempo Silhouette: 840.1551 s
"""

# ==========================================
# 2. PARSING DO MPI
# ==========================================
mpi_data = []
# Regex captura: N, K, Processos, Tempo Kmeans, Tempo Comm, Tempo Silhouette
pattern = re.compile(r"N=(\d+).*?P=(\d+).*?Tempo Total K-means: ([\d\.]+) s.*?Tempo Comunicação.*?: ([\d\.]+) s.*?Tempo Silhouette: ([\d\.]+) s", re.DOTALL)

matches = pattern.findall(mpi_log)
for match in matches:
    n, p, t_k, t_c, t_sil = match
    # Converter para ms para comparar com OpenMP/CUDA
    mpi_data.append({
        'n_pontos': int(n),
        'resources': int(p),
        'tempo': float(t_k) * 1000,     # seg -> ms
        'tempo_comm': float(t_c) * 1000, # seg -> ms
        'tempo_sil_s': float(t_sil),    # Mantém seg (é muito grande)
        'implementation': 'MPI',
        'resource_type': 'Processes'
    })

df_mpi = pd.DataFrame(mpi_data)
# Calcular Speedup MPI (baseado no P=1 de cada dataset)
baseline_mpi = df_mpi[df_mpi['resources'] == 1].set_index('n_pontos')['tempo']
df_mpi['speedup'] = df_mpi.apply(lambda row: baseline_mpi[row['n_pontos']] / row['tempo'] if row['tempo'] > 0 else 0, axis=1)

# ==========================================
# 3. CARREGAMENTO DOS CSVs (OpenMP e CUDA)
# ==========================================
# Simulação dos dados dos CSVs (baseado no que você enviou antes)
# Se estiver rodando local, use pd.read_csv('nome_do_arquivo.csv')

# --- CUDA Mock Data (Recriando a partir dos seus snippets) ---
cuda_raw = """
n_pontos,resources,tempo
10000,128,8.4
10000,256,7.5
10000,512,7.5
100000,128,1319.6
100000,256,1333.2
100000,512,1298.5
1000000,128,253719.6
1000000,256,253466.8
1000000,512,256464.0
"""
# Tente carregar do arquivo real se existir, senão usa o mock
try:
    df_cuda = pd.read_csv('results_cuda_with_speedup.csv')
    df_cuda = df_cuda.rename(columns={'block_size': 'resources'})
    df_cuda = df_cuda[['n_pontos', 'resources', 'tempo']]
except:
    df_cuda = pd.read_csv(io.StringIO(cuda_raw.strip()))

df_cuda['implementation'] = 'CUDA'
df_cuda['resource_type'] = 'Block Size'
# Speedup CUDA (Assumindo baseline serial hipotético ou relativo ao menor tempo)
# Como não tenho o serial CUDA no snippet, vou deixar speedup relativo a ele mesmo ou vazio
# Para o gráfico ficar bonito, vamos calcular speedup relativo ao pior caso CUDA (recurso menor) ou deixar 0
df_cuda['speedup'] = 1.0 # Placeholder se não tiver serial

# --- OpenMP Mock Data ---
omp_raw = """
n_pontos,n_centroids,iteracoes,tempo,serial_omp,n_threads,speedup
10000,4,3,95.3,1,1,1.0
10000,4,3,99.9,0,1,0.9539539539539539
10000,4,3,52.9,0,2,1.8015122873345937
10000,4,3,24.9,0,4,3.8273092369477912
10000,4,3,28.1,0,8,3.391459074733096
10000,4,3,15.4,0,16,6.188311688311688
10000,4,3,11.7,0,32,8.145299145299145
100000,8,3,16512.6,1,1,1.0
100000,8,3,16410.3,0,1,1.0062338896912304
100000,8,3,8348.8,0,2,1.977841126868532
100000,8,3,4171.7,0,4,3.9582424431287
100000,8,3,2115.3,0,8,7.806268614380937
100000,8,3,1715.4,0,16,9.626093039524308
100000,8,3,1355.1,0,32,12.185521363736994
1000000,16,3,2955223.8,1,1,1.0
1000000,16,3,3023151.2,0,1,0.9775309286548419
1000000,16,3,1548262.3,0,2,1.9087358776352041
1000000,16,3,771009.8,0,4,3.832926377848893
1000000,16,3,407779.6,0,8,7.247110448879738
1000000,16,3,373915.5,0,16,7.903453587775847
1000000,16,3,275368.1,0,32,10.731903223358117
"""
try:
    df_omp = pd.read_csv('results/results_with_speedup.csv')
    df_omp = df_omp.rename(columns={'n_threads': 'resources'})
    df_omp = df_omp[['n_pontos', 'resources', 'tempo', 'speedup']]
except:
    print("Usando dados mock para OpenMP")
    df_omp = pd.read_csv(io.StringIO(omp_raw.strip()))
    df_omp = df_omp.rename(columns={'n_threads': 'resources'})

df_omp['implementation'] = 'OpenMP'
df_omp['resource_type'] = 'Threads'
# Recalcular speedup OpenMP
baseline_omp = df_omp[df_omp['resources'] == 1].set_index('n_pontos')['tempo']
# Fallback se não tiver resources=1 para algum caso
def get_omp_speedup(row):
    if row['n_pontos'] in baseline_omp.index:
        return baseline_omp[row['n_pontos']] / row['tempo']
    return np.nan
# df_omp['speedup'] = df_omp.apply(get_omp_speedup, axis=1)

print("Dados OMP: ")
print(df_omp.head())
# ==========================================
# 4. MERGE E PLOTAGEM
# ==========================================
df_final = pd.concat([df_mpi, df_cuda, df_omp], ignore_index=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# --- GRÁFICO A: Comparação de Tempo Absoluto (Escala Log) ---
# Mostra quem é mais rápido de fato
sns.barplot(data=df_final, x="n_pontos", y="tempo", hue="implementation", 
            palette="viridis", ax=axes[0, 0])
axes[0, 0].set_title("Tempo Total de Execução (Menor é Melhor)")
axes[0, 0].set_ylabel("Tempo (ms) - Escala Log")
axes[0, 0].set_yscale("log")
axes[0, 0].legend(title="Implementação")

# --- GRÁFICO B: Speedup Relativo (Escalabilidade) ---
# Mostra quem aproveita melhor os recursos adicionais
# Filtrando CUDA pois BlockSize não é comparável linearmente com Threads/Procs no eixo X
df_scaling = df_final[df_final['implementation'].isin(['MPI', 'OpenMP'])]
max_res = df_scaling['resources'].max()
axes[0, 1].plot([1, max_res], [1, max_res], '--', color='gray', alpha=0.5, label='Ideal Linear')
sns.lineplot(data=df_scaling, x="resources", y="speedup", hue="n_pontos", style="implementation",
             markers=True, dashes=False, linewidth=2.5, palette="tab10", ax=axes[0, 1])
# Adiciona linha ideal
axes[0, 1].set_title("Speedup: MPI vs OpenMP")
axes[0, 1].set_ylabel("Speedup (x)")
axes[0, 1].set_xlabel("Número de Recursos (Threads/Processos)")
axes[0, 1].set_xscale('log', base=2)
axes[0, 1].set_xticks([1, 2, 4, 8, 16, 32])
axes[0, 1].set_xticklabels([1, 2, 4, 8, 16, 32])

# --- GRÁFICO C: Análise de Custo MPI (Comunicação) ---
# Mostra o overhead matando a performance em datasets pequenos
sns.barplot(data=df_mpi, x="resources", y="tempo", hue="n_pontos", ax=axes[1, 0])
# Sobrepor o tempo de comunicação
# Truque: Plotar tempo total e tempo de comunicação para ver a proporção
axes[1, 0].set_title("MPI: Tempo Total vs Recursos (Gargalo em Pequenos Datasets)")
axes[1, 0].set_ylabel("Tempo (ms)")
axes[1, 0].set_yscale("log")

# --- GRÁFICO D: O Desafio do Silhouette (MPI) ---
# Focando no Silhouette que é pesado
sns.lineplot(data=df_mpi, x="resources", y="tempo_sil_s", hue="n_pontos", 
             marker="o", palette="tab20", ax=axes[1, 1])
axes[1, 1].set_title("Custo Computacional do Silhouette (MPI)")
axes[1, 1].set_ylabel("Tempo (segundos)")
axes[1, 1].set_xlabel("Número de Processos")
axes[1, 1].set_yscale("log")
axes[1, 1].grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.savefig('analise_comparativa_pcd.png', dpi=300)
print("Gráficos gerados com sucesso: 'analise_comparativa_pcd.png'")