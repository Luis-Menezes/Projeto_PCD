#!/bin/bash

# filepath: /home/luis-menezes/Documents/Grad/2025_PCD/Projeto_PCD/run_tests.sh

# Run tests for OpenMP implementations with different scheduling strategies

mkdir -p results_sched

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="results_sched/test_results_${TIMESTAMP}.txt"

# Test configurations:
# We will focus on the largest dataset for scheduling tests
CONFIGS="1000000 16"


THREAD_COUNTS=(32)

CHUNK_SIZES=(0 10 100 1000)

SCHEDULE_TYPES=("static" "dynamic")


echo "Executando testes completos para implementação serial e OpenMP"
echo "Resultados serão salvos em: $OUTPUT_FILE"
echo ""

log() {
    echo "$1" | tee -a "$OUTPUT_FILE"
}

log "Compiling only the new opMP code for scheduling tests..."
gcc -std=c99 data/geradorDados.c -o data/geradorDados
gcc -std=c99 data/geradorCentroides.c -o data/geradorCentroides
gcc -O2 -std=c99 -fopenmp openMP/opMP_schedule.c -o openMP/kmeans_omp_sched -lm
log "Compilação concluída!"
log ""


echo "=== K-MEANS 1D SCHEDULE TEST RESULTS ===" > "$OUTPUT_FILE"
echo "Data e hora: $(date)" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"


read N_DATA_POINTS K_CENTROIDS <<< "$CONFIGS"
log "Gerando dados: $N_DATA_POINTS pontos e $K_CENTROIDS centróides iniciais"
./data/geradorDados $N_DATA_POINTS $K_CENTROIDS
./data/geradorCentroides $K_CENTROIDS
log ""

for schedule in "${SCHEDULE_TYPES[@]}"; do
    for threads in "${THREAD_COUNTS[@]}"; do
            for chunk_size in "${CHUNK_SIZES[@]}"; do
                
                log "========================================"
                log "TESTANDO: $N_DATA_POINTS pontos, $K_CENTROIDS clusters, Schedule: $schedule, Threads: $threads Chunk Sizes: ${chunk_size[*]}"
                log "========================================"
                


                log "Executando k-means OpenMP com agendamento $schedule, chunk size de $chunk_size e $threads threads..."

                ./openMP/kmeans_omp data/dados.csv data/centroides_iniciais.csv 50 0.000001 $threads $chunk_size openMP/assign_${N_DATA_POINTS}_${threads}t.csv openMP/centroids_${N_DATA_POINTS}_${threads}t.csv 2>&1 | tee -a "$OUTPUT_FILE"


                # log "Resultado da execução:"
                log ""
            done
    done
done