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

if __name__ == "__main__":
    path = 'results/results_cuda.csv'
    if not Path(path).exists():
        print(f"File {path} not found")
    else:
        df = load_and_prepare(path)
        df = calculate_speedup(df)
        df.to_csv('results/results_cuda_with_speedup.csv', index=False)
        print_summary(df)
        plot_speedup_by_block(df)

