#!/bin/bash

# filepath: /home/luis-menezes/Documents/Grad/2025_PCD/Projeto_PCD/run_tests.sh

# Create results directory if it doesn't exist
mkdir -p results

# Set output file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="results/test_results_${TIMESTAMP}.txt"

N_DATA_POINTS=10000
K_CENTROIDS=4
N_THREADS=4

echo "Gerando um teste para implementação serial e OpenMP"
echo "Resultados serão salvos em: $OUTPUT_FILE"
echo ""

# Function to log to both console and file
log() {
    echo "$1" | tee -a "$OUTPUT_FILE"
}

# Clear/create the output file
echo "=== K-MEANS 1D TEST RESULTS ===" > "$OUTPUT_FILE"
echo "Data e hora: $(date)" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

log "Gerando dados e centróides iniciais"
log "Gerando $N_DATA_POINTS pontos de dados e $K_CENTROIDS centróides iniciais"
log ""

gcc -std=c99 data/geradorDados.c -o data/geradorDados
gcc -std=c99 data/geradorCentroides.c -o data/geradorCentroides
./data/geradorDados $N_DATA_POINTS $K_CENTROIDS
./data/geradorCentroides $K_CENTROIDS

log "Compilando ambos os códigos"

# Compile serial version
gcc -O2 -std=c99 serial/kmeans_1d_serial.c -o serial/kmeans_1d_naive -lm

# Compile OpenMP version
gcc -O2 -std=c99 -fopenmp openMP/opMP.c -o openMP/kmeans_omp -lm

log "Executando ambos os códigos com os mesmos dados de entrada"
log ""
log "=== IMPLEMENTAÇÃO SERIAL ==="
./serial/kmeans_1d_naive data/dados.csv data/centroides_iniciais.csv 50 0.000001 $N_THREADS serial/assign.csv serial/centroids.csv 2>&1 | tee -a "$OUTPUT_FILE"

log ""
log "=== IMPLEMENTAÇÃO OPENMP ==="
./openMP/kmeans_omp data/dados.csv data/centroides_iniciais.csv 50 0.000001 $N_THREADS openMP/assign.csv openMP/centroids.csv 2>&1 | tee -a "$OUTPUT_FILE"

log ""
log "=== COMPARAÇÃO DE RESULTADOS ==="

# Compare centroid results
if cmp -s serial/centroids.csv openMP/centroids.csv; then
    log "✓ Centróides finais são idênticos entre as implementações"
else
    log "⚠ Diferenças encontradas nos centróides finais"
fi

# Compare assignments
if cmp -s serial/assign.csv openMP/assign.csv; then
    log "✓ Atribuições são idênticas entre as implementações"
else
    log "⚠ Diferenças encontradas nas atribuições"
fi

log ""
log "=== INFORMAÇÕES DO SISTEMA ==="
log "CPU: $(cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d':' -f2 | xargs)"
log "Cores disponíveis: $(nproc)"
log "Threads OpenMP utilizadas: ${OMP_NUM_THREADS:-$(nproc)}"
log ""
log "Teste concluído. Resultados salvos em: $OUTPUT_FILE"

echo ""
echo "Teste concluído. Resultados salvos em: $OUTPUT_FILE"