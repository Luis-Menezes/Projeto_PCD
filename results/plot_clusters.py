import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os

def plot_clusters(data_file='data/dados.csv', 
                  labels_file='serial/output', 
                  centroids_file=None,
                  output_file='results/figures/clusters_visualization_10k.png'):
    """
    Plota os pontos coloridos por cluster com os centroides marcados
    
    Args:
        data_file: arquivo com os dados (um valor por linha)
        labels_file: arquivo com os labels de cluster (um por linha)
        centroids_file: arquivo com os centroides finais (opcional, auto-detecta)
        output_file: onde salvar o gráfico
    """
    
    # 1. Ler os dados
    try:
        data = pd.read_csv(data_file, header=None, names=['value'])
        print(f"✓ Dados carregados: {len(data)} pontos")
    except FileNotFoundError:
        print(f"ERRO: Arquivo {data_file} não encontrado!")
        return
    
    # 2. Ler os labels (clusters atribuídos)
    try:
        labels = pd.read_csv(labels_file, header=None, names=['cluster'])
        print(f"✓ Labels carregados: {len(labels)} clusters")
    except FileNotFoundError:
        print(f"ERRO: Arquivo {labels_file} não encontrado!")
        return
    
    # Verificar compatibilidade
    if len(data) != len(labels):
        print(f"AVISO: Número de pontos ({len(data)}) != número de labels ({len(labels)})")
        min_len = min(len(data), len(labels))
        data = data.iloc[:min_len]
        labels = labels.iloc[:min_len]
    
    # 3. Auto-detectar arquivo de centroides se não fornecido
    if centroids_file is None:
        # Procura por centroids_N.csv onde N é o tamanho do dataset
        N = len(data)
        centroids_file = f'centroids_{N}.csv'
        if not os.path.exists(centroids_file):
            # Tenta em serial/
            centroids_file = f'serial/centroids_{N}.csv'
    
    # 4. Ler os centroides
    centroids = None
    if centroids_file and os.path.exists(centroids_file):
        try:
            centroids = pd.read_csv(centroids_file, header=None, names=['value'])
            print(f"✓ Centroides carregados: {len(centroids)} clusters")
        except Exception as e:
            print(f"AVISO: Não foi possível ler centroides de {centroids_file}: {e}")
    else:
        print(f"AVISO: Arquivo de centroides não encontrado: {centroids_file}")
    
    # 5. Combinar dados e labels
    df = pd.DataFrame({
        'index': range(len(data)),
        'value': data['value'].values,
        'cluster': labels['cluster'].values
    })
    
    # 6. Criar o gráfico
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Número de clusters únicos
    K = df['cluster'].nunique()
    palette = sns.color_palette("husl", n_colors=K)
    
    # Plotar pontos coloridos por cluster
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        ax.scatter(cluster_data['index'], cluster_data['value'], 
                  c=[palette[cluster_id]], label=f'Cluster {cluster_id}',
                  alpha=0.6, s=20)
    
    # Plotar centroides (linhas horizontais)
    if centroids is not None:
        for i, centroid_val in enumerate(centroids['value']):
            ax.axhline(y=centroid_val, color=palette[i % K], 
                      linestyle='--', linewidth=2, alpha=0.8)
            # Adicionar label do centroide
            ax.text(len(df) * 0.02, centroid_val, f'C{i}', 
                   bbox=dict(boxstyle='round', facecolor=palette[i % K], alpha=0.7),
                   fontsize=9, verticalalignment='center')
    
    # Formatação
    ax.set_xlabel('Índice do Ponto', fontsize=12)
    ax.set_ylabel('Valor', fontsize=12)
    ax.set_title(f'Visualização dos Clusters (N={len(df):,}, K={K})', fontsize=14)
    ax.legend(loc='upper right', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Salvar
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"✓ Gráfico salvo em: {output_file}")
    plt.show()
    
    # Estatísticas
    print("\n=== Estatísticas por Cluster ===")
    stats = df.groupby('cluster')['value'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(stats)
    
    if centroids is not None:
        print("\n=== Centroides Finais ===")
        for i, val in enumerate(centroids['value']):
            print(f"Cluster {i}: {val:.4f}")


if __name__ == "__main__":
    # Aceita argumentos da linha de comando
    if len(sys.argv) > 1:
        data_file = sys.argv[1] if len(sys.argv) > 1 else 'data/dados.csv'
        labels_file = sys.argv[2] if len(sys.argv) > 2 else 'serial/output'
        centroids_file = sys.argv[3] if len(sys.argv) > 3 else None
        output_file = sys.argv[4] if len(sys.argv) > 4 else 'results/figures/clusters_visualization_10k.png'
        
        plot_clusters(data_file, labels_file, centroids_file)
    else:
        # Uso padrão
        print("Uso: python plot_clusters.py [dados.csv] [output] [centroids_N.csv]")
        print("Executando com arquivos padrão...\n")
        plot_clusters()