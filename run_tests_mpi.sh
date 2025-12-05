#!/bin/bash

# filepath: run_tests_mpi.sh

# Cria diretórios necessários para organização
mkdir -p results
mkdir -p mpi/output
mkdir -p serial/output

# Define arquivo de saída com timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="results/mpi_test_results_${TIMESTAMP}.txt"

# Configurações de teste: N_DATA_POINTS K_CENTROIDS
# (Nota: Removi o BLOCK_SIZE pois ele é específico de CUDA, não costuma usar em MPI)
CONFIGS=(
    "10000 4"
    "100000 8"
    "1000000 16"
)

# Quantidade de processos MPI a serem testados
PROCESS_COUNTS=(1 2 4 8 16 32)

echo "Executando testes completos para implementação MPI"
echo "Resultados serão salvos em: $OUTPUT_FILE"
echo ""

# Função de log
log() {
    echo "$1" | tee -a "$OUTPUT_FILE"
}

# Cabeçalho do arquivo
echo "=== K-MEANS MPI COMPREHENSIVE TEST RESULTS ===" > "$OUTPUT_FILE"
echo "Data e hora: $(date)" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

log "=== INFORMAÇÕES DO SISTEMA ==="
log "CPU: $(cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d':' -f2 | xargs)"
log "Cores Físicos: $(grep -c ^processor /proc/cpuinfo)"
log "MPI Compiler: $(mpicc --version | head -1)"
log ""

# --- COMPILAÇÃO ---
log "Compilando os códigos..."

# 1. Geradores de dados (GCC padrão)
gcc -std=c99 data/geradorDados.c -o data/geradorDados
gcc -std=c99 data/geradorCentroides.c -o data/geradorCentroides

# 2. Versão Serial (Para comparação/validação) - GCC padrão
# Certifique-se que o caminho do seu serial está correto
gcc -O2 -std=c99 serial/kmeans_1d_serial.c -o serial/kmeans_serial -lm 

# 3. Versão MPI (Wrapper MPICC)
# Ajuste o nome do arquivo .c se necessário
mpicc -O2 -std=c99 mpi/kmeans_mpi.c -o mpi/kmeans_mpi -lm

log "Compilação concluída!"
log ""

# --- LOOP PRINCIPAL DE TESTES ---
for config in "${CONFIGS[@]}"; do
    read N_DATA_POINTS K_CENTROIDS <<< "$config"
    
    log "========================================"
    log "DATASET: $N_DATA_POINTS pontos, $K_CENTROIDS clusters"
    log "========================================"
    
    # Gera dados novos para esta configuração
    log ">> Gerando dados..."
    ./data/geradorDados $N_DATA_POINTS $K_CENTROIDS
    ./data/geradorCentroides $K_CENTROIDS
    
    # 1. Executa Serial (Base para comparação)
    log ">> Executando Serial (Gabarito)..."
    # ./serial/kmeans_serial data/dados.csv data/centroides_iniciais.csv 50 0.000001 serial/output/assign_serial.csv serial/output/centroids_serial.csv > /dev/null
    
    log ""
    
    # 2. Loop variando número de processos MPI
    for P in "${PROCESS_COUNTS[@]}"; do
        log "--- MPI com $P processos ---"
        
        # Arquivos de saída específicos para cada contagem de processos para não sobrescrever
        OUT_ASSIGN="mpi/output/assign_${N_DATA_POINTS}_P${P}.csv"
        OUT_CENTROIDS="mpi/output/centroids_${N_DATA_POINTS}_P${P}.csv"
        
        # Execução do MPI
        # O parâmetro --oversubscribe permite rodar mais processos que cores físicos (útil para testar 32 procs em PC pessoal)
        mpirun --oversubscribe -np $P ./mpi/kmeans_mpi data/dados.csv data/centroides_iniciais.csv 50 0.000001 $OUT_ASSIGN $OUT_CENTROIDS 2>&1 | tee -a "$OUTPUT_FILE"
        
        # Validação dos resultados (Compara com Serial)
        # Nota: K-means é sensível a ponto flutuante, 'cmp' pode falhar por diferenças mínimas.
        # Se falhar muito, considere criar um script python para comparar com tolerância.
        if cmp -s serial/output/centroids_serial.csv $OUT_CENTROIDS; then
            log "   [OK] Centróides conferem com Serial."
        else
            log "   [FAIL] Centróides DIFEREM da versão Serial."
        fi
        
        log ""
    done
    
    log "Configuração $N_DATA_POINTS finalizada."
    log ""
done

log "Teste completo concluído. Resultados em: $OUTPUT_FILE"