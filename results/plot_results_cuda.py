import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os

def load_and_prepare(path='results/results_cuda.csv'):
    df = pd.read_csv(path, skipinitialspace=True)
    # normalize columns
    df.columns = df.columns.str.strip()
    # ensure types
    df['n_pontos'] = df['n_pontos'].astype(int)
    df['n_centroids'] = df['n_centroids'].astype(int)
    df['tempo'] = df['tempo'].astype(float)
    df['serial_cuda'] = df['serial_cuda'].astype(int)
    df['block_size'] = df['block_size'].astype(int)
    return df

def calculate_speedup(df):
    df2 = df.copy()
    df2['speedup'] = np.nan
    for (n, k), g in df.groupby(['n_pontos','n_centroids']):
        serial = g[g['serial_cuda']==1]
        if serial.empty:
            # no serial baseline, skip
            continue
        serial_time = float(serial['tempo'].iloc[0])
        mask = (df2['n_pontos']==n) & (df2['n_centroids']==k)
        df2.loc[mask, 'speedup'] = serial_time / df2.loc[mask,'tempo']
    return df2

def plot_speedup_by_block(df, out='results/figures/speedup_cuda.png'):
    sns.set_theme(style="whitegrid")
    os.makedirs(Path(out).parent, exist_ok=True)

    # prepare data: only entries with speedup computed
    dfp = df.dropna(subset=['speedup']).copy()
    datasets = sorted(dfp[['n_pontos','n_centroids']].drop_duplicates().values.tolist())

    plt.figure(figsize=(8,5))
    for n,k in datasets:
        sub = dfp[(dfp['n_pontos']==n)&(dfp['n_centroids']==k)&(dfp['serial_cuda']==0)]
        if sub.empty: 
            continue
        sub = sub.sort_values('block_size')
        plt.plot(sub['block_size'], sub['speedup'], '-o', label=f'{n:,} pts, K={k}')
    plt.xlabel('Block size')
    plt.ylabel('Speedup (serial_time / gpu_time)')
    plt.title('CUDA K-means: Speedup vs block size')
    plt.legend()
    plt.ylim(bottom=0, top=15)
    plt.xscale('log', base=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.show()

def print_summary(df):
    print("=== CUDA Results Summary ===")
    for (n,k), g in df.groupby(['n_pontos','n_centroids']):
        serial = g[g['serial_cuda']==1]
        if serial.empty: 
            print(f"{n:,} pts K={k}: serial baseline missing")
            continue
        serial_time = serial['tempo'].iloc[0]
        best = g[g['serial_cuda']==0].sort_values('tempo').head(1)
        if not best.empty:
            print(f"{n:,} pts K={k}: serial {serial_time:.3f} ms | best GPU {best['tempo'].iloc[0]:.3f} ms (block {int(best['block_size'].iloc[0])}) | speedup {serial_time/best['tempo'].iloc[0]:.3f}x")
        else:
            print(f"{n:,} pts K={k}: only serial runs")

def plot_comparative_results():
    import matplotlib.pyplot as plt
    import numpy as np

    # --- 1. Dados extraídos dos seus logs (usando threadsPerBlock=256) ---

    # Rótulos para o eixo X
    labels = ['10k pts, K=4', '100k pts, K=8', '1M pts, K=16']

    # Tempos de transferência Host-para-Device (H2D)
    h2d_times = np.array([0.0, 0.0, 0.0])

    # Tempos de execução do Kernel na GPU
    kernel_times = np.array([0.1, 0.2, 1.4])

    # Tempos de transferência Device-para-Host (D2H)
    d2h_times = np.array([0.1, 0.8, 5.7])

    # Calcula os tempos totais para adicionar rótulos no topo
    totals = h2d_times + kernel_times + d2h_times

    # Largura das barras
    width = 0.5

    # --- 2. Criação do Gráfico Empilhado ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # A. A primeira barra (base) é o H2D
    ax.bar(labels, h2d_times, width, label='Tempo H2D (ms)', color='#2c7bb6')

    # B. A segunda barra (Kernel) começa onde a H2D terminou
    ax.bar(labels, kernel_times, width, bottom=h2d_times,
        label='Tempo Kernel (ms)', color='#abd9e9')

    # C. A terceira barra (D2H) começa onde a Kernel terminou
    ax.bar(labels, d2h_times, width, bottom=h2d_times + kernel_times,
        label='Tempo D2H (ms)', color='#fdae61')

    # --- 3. Formatação e Rótulos ---

    # Títulos e rótulos dos eixos
    ax.set_ylabel('Tempo Total (ms)')
    ax.set_title('Breakdown do Tempo K-means (GPU) por DBK (threadPerBlock=256)', fontsize=16)

    # Adiciona a legenda
    ax.legend(loc='upper left')

    # Adiciona grade para facilitar a leitura
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Define o limite do eixo Y um pouco mais alto que o valor máximo
    ax.set_ylim(0, np.max(totals) * 1.15)

    # Adiciona rótulos de texto com o valor total no topo de cada barra
    for i, total in enumerate(totals):
        if total > 0: # Só adiciona rótulo se o tempo for maior que 0
            ax.text(i, total + (np.max(totals) * 0.01), # Posição Y (um pouco acima da barra)
                    f'{total:.1f} ms',                 # Texto do rótulo
                    ha='center',                       # Alinhamento horizontal
                    fontweight='bold')

    # Melhora o layout para evitar cortes
    plt.tight_layout()

    # --- 4. Salvar e Mostrar o Gráfico ---
    plt.savefig('results/figures/kmeans_breakdown_tempo_gpu.png')
    plt.show()


def format_n(n):
        if n >= 1000000:
            return f"{n // 1000000}M"
        if n >= 1000:
            return f"{n // 1000}k"
        return str(n)

def plot_comparative_results_speedup():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # --- 1. Carregar e Preparar os Dados ---

    try:
        # Carrega os dados de OpenMP e Serial
        df_omp_serial = pd.read_csv('results/results_with_speedup.csv')
        
        # Carrega os dados de CUDA
        df_cuda = pd.read_csv('results/results_cuda_with_speedup.csv')
    except FileNotFoundError as e:
        print(f"Erro: Não foi possível encontrar o arquivo {e.filename}.")
        print("Certifique-se que os arquivos 'results_with_speedup.csv' e 'results_cuda_with_speedup.csv' estão no mesmo diretório.")
        exit()

    # --- 2. CORREÇÃO: Criar a coluna 'Implementation' ---

    # Em df_omp_serial, use 'serial_omp' para definir 'Serial' ou 'OpenMP'
    df_omp_serial['Implementation'] = np.where(df_omp_serial['serial_omp'] == 1, 'Serial', 'OpenMP') # <<< NOVO

    # Em df_cuda, todas as linhas são 'CUDA'
    df_cuda['Implementation'] = 'CUDA' # <<< NOVO

    # --- 3. CORREÇÃO: Padronizar nomes das colunas ---
    # Renomeia 'n_pontos' -> 'N' e 'n_centroids' -> 'K' para ambos os DataFrames
    df_omp_serial.rename(columns={'n_pontos': 'N', 'n_centroids': 'K'}, inplace=True) # <<< NOVO
    df_cuda.rename(columns={'n_pontos': 'N', 'n_centroids': 'K'}, inplace=True)       # <<< NOVO


    # --- 4. Juntar e Filtrar ---

    # Agora podemos juntar os DataFrames
    df_all = pd.concat([df_omp_serial, df_cuda], ignore_index=True)

    # Filtra apenas as implementações principais que queremos comparar
    implementations_to_plot = ['Serial', 'OpenMP', 'CUDA']
    df_plot = df_all[df_all['Implementation'].isin(implementations_to_plot)]

    # --- 5. Criar Rótulos para o Eixo X ---

    # Função para formatar N de forma legível (ex: 10000 -> 10k, 1000000 -> 1M)
    def format_n(n):
        if n >= 1000000:
            return f"{n // 1000000}M"
        if n >= 1000:
            return f"{n // 1000}k"
        return str(n)

    # Cria uma coluna 'Config' para usar como rótulo no eixo X
    # Adiciona .copy() para evitar o SettingWithCopyWarning
    df_plot = df_plot.copy()
    df_plot['Config'] = df_plot['N'].apply(format_n) + ' pts, K=' + df_plot['K'].astype(str)

    # Garante que os dados estejam ordenados por N (tamanho do problema)
    df_plot = df_plot.sort_values(by='N')

    # --- 6. Configuração do Gráfico de Barras Agrupado ---

    # Pega os rótulos únicos de configuração (ex: "10k...", "100k...", "1M...")
    configs = df_plot['Config'].unique()
    x_indexes = np.arange(len(configs))  # Posições no eixo X (0, 1, 2)

    # Define a largura de cada barra individual
    bar_width = 0.25
    n_impls = len(implementations_to_plot)

    # Cores para consistência
    colors = {
        'Serial': '#d73027', # Vermelho
        'OpenMP': '#fee090', # Amarelo/Laranja
        'CUDA': '#4575b4'    # Azul
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    # --- 7. Plotar as Barras ---

    # Itera sobre cada implementação para plotar suas barras em grupo
    for i, impl in enumerate(implementations_to_plot):
        # Calcula o "offset" da barra para que fiquem lado a lado
        offset = (i - n_impls / 2 + 0.5) * bar_width
        positions = x_indexes + offset
        
        # Pega os dados de speedup para a implementação atual
        # Precisamos agrupar e pegar a média caso haja múltiplas entradas (ex: CUDA com 128, 256, 512)
        speedups = df_plot[df_plot['Implementation'] == impl].groupby('Config', sort=False)['Speedup'].mean().values
        
        # Plota a barra
        ax.bar(positions, speedups, bar_width, label=impl, color=colors.get(impl))

    # --- 8. Formatação Final do Gráfico ---

    # Títulos e Rótulos
    ax.set_ylabel('Speedup (Base: Tempo Serial)')
    ax.set_title('Comparativo de Speedup: Serial vs. OpenMP vs. CUDA', fontsize=16)

    # Configura os rótulos do eixo X para ficarem no centro do grupo de barras
    ax.set_xticks(x_indexes)
    ax.set_xticklabels(configs)

    # Adiciona uma linha horizontal em y=1.0 para marcar a base (Serial)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Baseline Serial (Speedup=1)')

    # Adiciona grade e legenda
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title="Implementação")

    # Otimiza o layout e salva a imagem
    plt.tight_layout()
    plt.savefig('comparativo_speedup_total.png')

    print("Gráfico 'comparativo_speedup_total.png' gerado com sucesso!")

    # Mostra o gráfico
    plt.show()

if __name__ == "__main__":
    # path = 'results/results_cuda.csv'
    # if not Path(path).exists():
    #     print(f"File {path} not found")
    # else:
    #     df = load_and_prepare(path)
    #     df = calculate_speedup(df)
    #     df.to_csv('results/results_cuda_with_speedup.csv', index=False)
    #     print_summary(df)
    #     plot_speedup_by_block(df)

    plot_comparative_results_speedup()
