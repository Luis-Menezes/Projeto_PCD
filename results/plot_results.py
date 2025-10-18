import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



def calculate_speedup(df):
    """
    Calcula o speedup comparando com a versão serial (n_threads=1)
    """
    df_with_speedup = df.copy()
    df_with_speedup['speedup'] = 1.0  # Inicializa com 1.0
    
    # Para cada configuração (n_pontos, n_centroids), calcula o speedup
    for (n_pontos, n_centroids), group in df.groupby(['n_pontos', 'n_centroids']):
        # Encontra o tempo da versão serial (n_threads=1 ou serial_omp=True)
        serial_time = group[group['n_threads'] == 1]['tempo'].iloc[0]
        
        # Calcula speedup para todos os casos dessa configuração
        for idx in group.index:
            parallel_time = df.loc[idx, 'tempo']
            df_with_speedup.loc[idx, 'speedup'] = serial_time / parallel_time
    
    return df_with_speedup

def plot_speedup(df):
    """
    Plot speedup vs number of threads for different dataset sizes
    """
    sns.set(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Speedup vs Threads
    df_parallel = df[df['serial_omp'] == False]  # Apenas versões paralelas
    
    for (n_pontos, n_centroids), group in df_parallel.groupby(['n_pontos', 'n_centroids']):
        ax1.plot(group['n_threads'], group['speedup'], 
                marker='o', linewidth=2, markersize=8,
                label=f'{n_pontos:,} pontos, {n_centroids} clusters')
    
    # Linha ideal (speedup linear)
    max_threads = df['n_threads'].max()
    ax1.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5, label='Speedup ideal')
    
    ax1.set_xlabel('Número de Threads')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Speedup vs Número de Threads')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=2)
    
    # Plot 2: Eficiência vs Threads
    df_parallel['efficiency'] = df_parallel['speedup'] / df_parallel['n_threads']
    
    for (n_pontos, n_centroids), group in df_parallel.groupby(['n_pontos', 'n_centroids']):
        ax2.plot(group['n_threads'], group['efficiency'], 
                marker='s', linewidth=2, markersize=8,
                label=f'{n_pontos:,} pontos, {n_centroids} clusters')
    
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Eficiência ideal')
    ax2.set_xlabel('Número de Threads')
    ax2.set_ylabel('Eficiência')
    ax2.set_title('Eficiência vs Número de Threads')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig('results/figures/speedup_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_execution_time(df):
    """
    Plot execution time comparison
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Criar subplot para cada configuração
    configs = df.groupby(['n_pontos', 'n_centroids']).size().index
    
    fig, axes = plt.subplots(len(configs), 1, figsize=(12, 4*len(configs)))
    if len(configs) == 1:
        axes = [axes]
    
    for i, (n_pontos, n_centroids) in enumerate(configs):
        subset = df[(df['n_pontos'] == n_pontos) & (df['n_centroids'] == n_centroids)]
        
        # Separar serial e paralelo
        serial_data = subset[subset['serial_omp'] == True]
        parallel_data = subset[subset['serial_omp'] == False]
        
        # Plot tempo de execução
        if len(serial_data) > 0:
            axes[i].axhline(y=serial_data['tempo'].iloc[0], color='red', 
                           linestyle='--', linewidth=2, label='Serial')
        
        axes[i].plot(parallel_data['n_threads'], parallel_data['tempo'], 
                    'bo-', linewidth=2, markersize=8, label='OpenMP')
        
        axes[i].set_xlabel('Número de Threads')
        axes[i].set_ylabel('Tempo (ms)')
        axes[i].set_title(f'Tempo de Execução: {n_pontos:,} pontos, {n_centroids} clusters')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xscale('log', base=2)
        axes[i].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/figures/execution_time.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_speedup_summary(df):
    """
    Imprime um resumo dos resultados de speedup
    """
    print("=== RESUMO DE PERFORMANCE ===\n")
    
    for (n_pontos, n_centroids), group in df.groupby(['n_pontos', 'n_centroids']):
        print(f"Configuração: {n_pontos:,} pontos, {n_centroids} clusters")
        print("-" * 50)
        
        serial_time = group[group['n_threads'] == 1]['tempo'].iloc[0]
        print(f"Tempo serial: {serial_time:.1f} ms")
        
        parallel_group = group[group['serial_omp'] == False]
        for _, row in parallel_group.iterrows():
            speedup = row['speedup']
            efficiency = speedup / row['n_threads']
            print(f"  {row['n_threads']:2d} threads: {row['tempo']:8.1f} ms | "
                  f"Speedup: {speedup:5.2f}x | Eficiência: {efficiency:5.1%}")
        
        best_speedup = parallel_group['speedup'].max()
        best_threads = parallel_group.loc[parallel_group['speedup'].idxmax(), 'n_threads']
        print(f"  Melhor speedup: {best_speedup:.2f}x com {best_threads} threads")
        print()

if __name__ == "__main__":
    # Lê os dados
    df = pd.read_csv('results/results.csv')
    
    # Remove espaços em branco dos nomes das colunas
    df.columns = df.columns.str.strip()
    
    # Calcula o speedup
    df_with_speedup = calculate_speedup(df)
    
    # Salva o dataframe com speedup
    df_with_speedup.to_csv('results/results_with_speedup.csv', index=False)
    
    # Cria o diretório de figuras se não existir
    import os
    os.makedirs('results/figures', exist_ok=True)
    
    # Imprime resumo
    print_speedup_summary(df_with_speedup)
    
    # Gera os plots
    plot_speedup(df_with_speedup)
    plot_execution_time(df_with_speedup)
    
    print("Análise completa! Figuras salvas em results/figures/")


# def plot_results(df):
#     """
#     Plot the results for speedup time 
#     """
#     sns.set(style="whitegrid")
#     plt.figure(figsize=(10, 6))

#     # Example plot: Line plot of accuracy over epochs for different models
#     sns.lineplot(data=df, x='epoch', y='accuracy', hue='model', marker='o')

#     plt.title('Model Accuracy over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend(title='Model')
#     plt.tight_layout()
#     plt.savefig('results/figures/accuracy_plot.png')
#     plt.show()

# if __name__ == "__main__":
#     df = pd.read_csv('results/results.csv')
    
#     # Calcula o speed up
    
        
#     plot_results(df)

