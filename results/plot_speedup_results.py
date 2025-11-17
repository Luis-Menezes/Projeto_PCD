import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io # Para ler o CSV a partir de um texto

# --- 1. Dados CSV fornecidos pelo usuário ---
# Coloquei os dados do CSV que você colou em uma string.
# Isso evita problemas de 'Arquivo não encontrado'
csv_data = """n_pontos,n_centroids,iteracoes,tempo,serial_cuda,block_size,speedup
10000,4,3,8.4,0,128,11.345238095238095
10000,4,3,7.5,0,256,12.706666666666667
10000,4,3,7.5,0,512,12.706666666666667
100000,8,3,1319.6,0,128,12.51333737496211
100000,8,3,1333.2,0,256,12.385688568856883
100000,8,3,1298.5,0,512,12.71667308432807
1000000,16,3,253719.6,0,128,11.647597584104656
1000000,16,3,253466.8,0,256,11.65921454012912
1000000,16,3,256464.2,0,512,11.522948622068888
10000,4,3,95.3,1,1,1.0
100000,8,3,16512.6,1,1,1.0
1000000,16,3,2955223.8,1,1,1.0
10000,4,3,11.7,2,32,8.145299145299145
100000,8,3,1355.1,2,32,12.185521363736994
1000000,16,3,275368.1,2,32,10.731903223358117
"""

# --- 2. Carregar e Transformar os Dados ---

# Ler os dados da string
try:
    df = pd.read_csv(io.StringIO(csv_data))
except Exception as e:
    print(f"Erro ao ler os dados do CSV: {e}")
    exit()

# Criar a coluna 'Implementation' baseada na sua codificação
# 0 = CUDA, 1 = Serial, 2 = OpenMP
impl_map = {
    0: 'CUDA',
    1: 'Serial',
    2: 'OpenMP (32 th)' # Usei um nome mais descritivo
}
df['Implementation'] = df['serial_cuda'].map(impl_map)

# Função para formatar N (10000 -> 10k, 1000000 -> 1M)
def format_n(n):
    if n >= 1000000: return f"{n // 1000000}M"
    if n >= 1000: return f"{n // 1000}k"
    return str(n)

# Criar a coluna 'Config' para o eixo X
df['Config'] = df['n_pontos'].apply(format_n) + ' pts, K=' + df['n_centroids'].astype(str)

# --- 3. Filtrar os Dados para Plotagem ---

# Queremos:
# 1. Todas as entradas 'Serial'
is_serial = (df['Implementation'] == 'Serial')

# 2. Todas as entradas 'OpenMP (32 th)'
is_omp = (df['Implementation'] == 'OpenMP (32 th)')

# 3. APENAS as entradas 'CUDA' com block_size == 256 (como solicitado)
is_cuda_256 = (df['Implementation'] == 'CUDA') & (df['block_size'] == 256)

# Combinar os filtros
df_plot = df[is_serial | is_omp | is_cuda_256].copy()

# --- 4. Criar Tabela Pivot para Plotagem ---

# O 'pivot_table' é perfeito para agrupar
# Ele criará uma tabela com 'Config' nas linhas e 'Implementation' nas colunas
# Os valores que faltam (OMP para 100k e 1M no seu CSV) serão 'NaN', 
# que o matplotlib ignora automaticamente
df_pivot = df_plot.pivot_table(
    index='Config',
    columns='Implementation',
    values='speedup'
)

# Reordenar as colunas para uma ordem lógica no gráfico
# (Note que seu CSV só tem OMP para 10k, então só aparecerá lá)
plot_order = ['Serial', 'OpenMP (32 th)', 'CUDA']
df_pivot = df_pivot.reindex(columns=plot_order)

# Ordenar o índice (Config) pela ordem de N (10k, 100k, 1M)
# Extrai o número antes de 'k' ou 'M' e converte para int
sort_key = lambda x: int(''.join(filter(str.isdigit, x.split(' ')[0])))
df_pivot = df_pivot.reindex(sorted(df_pivot.index, key=sort_key))

# --- 5. Plotar o Gráfico de Barras Agrupado ---

print("Dados processados para plotagem (Speedup):")
print(df_pivot)
print("\nGerando gráfico...")

fig, ax = plt.subplots(figsize=(12, 7))

# O pandas faz o gráfico agrupado automaticamente com 'plot(kind='bar')'
df_pivot.plot(
    kind='bar',
    ax=ax,
    width=0.8,
    color={'Serial': '#d73027', 'OpenMP (32 th)': '#fee090', 'CUDA': '#4575b4'}
)

# --- 6. Formatação Final ---

ax.set_title('Comparativo de Speedup (CUDA com 256 th/bloco)', fontsize=16)
ax.set_ylabel('Speedup (Base: Tempo Serial)')
ax.set_xlabel('Configuração (N pontos, K clusters)')

# Rotacionar rótulos do eixo X para não sobrepor
plt.xticks(rotation=0)

# Adicionar linha de base
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Baseline Serial (Speedup=1)')

# Adicionar grade e legenda
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.legend(title="Implementação")

plt.tight_layout()
plt.savefig('comparativo_speedup_filtrado.png')

print("Gráfico 'comparativo_speedup_filtrado.png' gerado com sucesso!")

plt.show()