#!/bin/bash

# filepath: /home/luis-menezes/Documents/Grad/2025_PCD/Projeto_PCD/run_tests.sh

# Run execution and speedup tests for serial and OpenMP implementations
mkdir -p results

# Set output file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="results/test_results_${TIMESTAMP}.txt"

# Test configurations: N_DATA_POINTS K_CENTROIDS
CONFIGS=(
    "10000 4"
    "100000 8" 
    "1000000 16"
)

# Thread counts to test
THREAD_COUNTS=(1 2)

echo "Executando testes completos para implementação serial e OpenMP"
echo "Resultados serão salvos em: $OUTPUT_FILE"
echo ""

# Function to log to both console and file
log() {
    echo "$1" | tee -a "$OUTPUT_FILE"
}

# Clear/create the output file
echo "=== K-MEANS 1D COMPREHENSIVE TEST RESULTS ===" > "$OUTPUT_FILE"
echo "Data e hora: $(date)" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

log "=== INFORMAÇÕES DO SISTEMA ==="
log "CPU: $(cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d':' -f2 | xargs)"
log "Cores disponíveis: $(nproc)"
log "Compilador GCC: $(gcc --version | head -1)"
log ""

# Compile programs once
log "Compilando os códigos..."
gcc -std=c99 data/geradorDados.c -o data/geradorDados
gcc -std=c99 data/geradorCentroides.c -o data/geradorCentroides
gcc -O2 -std=c99 serial/kmeans_1d_serial.c -o serial/kmeans_1d_naive -lm
gcc -O2 -std=c99 -fopenmp openMP/opMP.c -o openMP/kmeans_omp -lm
log "Compilação concluída!"
log ""

# Main test loop
for config in "${CONFIGS[@]}"; do
    read N_DATA_POINTS K_CENTROIDS <<< "$config"
    
    log "========================================"
    log "TESTANDO: $N_DATA_POINTS pontos, $K_CENTROIDS clusters"
    log "========================================"
    
    # Generate data for this configuration
    log "Gerando dados: $N_DATA_POINTS pontos e $K_CENTROIDS centróides iniciais"
    ./data/geradorDados $N_DATA_POINTS $K_CENTROIDS
    ./data/geradorCentroides $K_CENTROIDS
    log ""
    
    # Test serial version (only once per dataset)
    # log "--- IMPLEMENTAÇÃO SERIAL ---"
    # ./serial/kmeans_1d_naive data/dados.csv data/centroides_iniciais.csv 50 0.000001 serial/assign_${N_DATA_POINTS}.csv serial/centroids_${N_DATA_POINTS}.csv 2>&1 | tee -a "$OUTPUT_FILE"
    # log ""
    
    # Test OpenMP with different thread counts
    log "--- IMPLEMENTAÇÃO OPENMP (variando threads) ---"
    
    for threads in "${THREAD_COUNTS[@]}"; do
        log ">> Testando com $threads thread(s):"
        ./openMP/kmeans_omp data/dados.csv data/centroides_iniciais.csv 50 0.000001 $threads openMP/assign_${N_DATA_POINTS}_${threads}t.csv openMP/centroids_${N_DATA_POINTS}_${threads}t.csv 2>&1 | tee -a "$OUTPUT_FILE"
        
        # Compare results with serial version
        if cmp -s serial/centroids_${N_DATA_POINTS}.csv openMP/centroids_${N_DATA_POINTS}_${threads}t.csv; then
            log "   ✓ Centróides idênticos à versão serial"
        else
            log "   ⚠ Diferenças nos centróides comparado à versão serial"
        fi
        
        if cmp -s serial/assign_${N_DATA_POINTS}.csv openMP/assign_${N_DATA_POINTS}_${threads}t.csv; then
            log "   ✓ Atribuições idênticas à versão serial"
        else
            log "   ⚠ Diferenças nas atribuições comparado à versão serial"
        fi
        log ""
    done
    
    log "Configuração $N_DATA_POINTS/$K_CENTROIDS concluída!"
    log ""
done

# Performance summary
log "========================================"
log "RESUMO DOS TESTES"
log "========================================"
log "Configurações testadas:"
for config in "${CONFIGS[@]}"; do
    read N K <<< "$config"
    log "  - $N pontos, $K clusters"
done
log ""
log "Threads testadas: ${THREAD_COUNTS[*]}"
log ""
log "Arquivos de saída gerados:"
log "  - Serial: serial/centroids_*.csv, serial/assign_*.csv"
log "  - OpenMP: openMP/centroids_*_*t.csv, openMP/assign_*_*t.csv"
log ""
log "Teste completo concluído. Resultados salvos em: $OUTPUT_FILE"

echo ""
echo "========================================" 
echo "Teste completo concluído!"
echo "Resultados salvos em: $OUTPUT_FILE"
echo "========================================"