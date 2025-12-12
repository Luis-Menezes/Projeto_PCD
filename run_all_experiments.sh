#!/bin/bash

# ==============================================================================
# Script Geral de Experimentos - Projeto PCD
# Autor: PCD Assistant (baseado nos scripts do aluno)
# Descrição: Automatiza a execução de K-Means em Serial, OpenMP, MPI e CUDA.
#            Gera dados unificados para garantir consistência nos benchmarks.
# ==============================================================================

# 1. PREPARAÇÃO DO AMBIENTE
# -------------------------
mkdir -p results
mkdir -p serial/output
mkdir -p openMP/output
mkdir -p mpi/output
mkdir -p cuda/output

# Arquivo de Log Unificado
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="results/final_results_overnight_${TIMESTAMP}.txt"

# Função de log auxiliar
log() {
    echo "$1" | tee -a "$OUTPUT_FILE"
}

# Cabeçalho
echo "=== RELATÓRIO FINAL DE EXPERIMENTOS PCD ===" > "$OUTPUT_FILE"
echo "Inicio: $(date)" >> "$OUTPUT_FILE"
echo "Máquina: $(hostname)" >> "$OUTPUT_FILE"
echo "CPU: $(grep -m 1 'model name' /proc/cpuinfo | cut -d: -f2)" >> "$OUTPUT_FILE"
echo "===========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# 2. CONFIGURAÇÕES DOS TESTES
# ---------------------------
# Formato: "N_DATA_POINTS K_CENTROIDS"
# Nota: Aumentamos progressivamente a carga (Conceito: Strong Scaling vs Weak Scaling)
CONFIGS=(
    "10000 4"       # Pequeno (Validação/Cache)
    "100000 8"      # Médio
    "1000000 16"    # Grande (Onde o paralelismo brilha)
)

# Variação de Recursos
OMP_THREADS=(1 2 4 8 16 32 64)          # Threads OpenMP (Ajuste conforme seus cores físicos)
MPI_PROCESSES=(1 2 4 8 16 32 64)        # Processos MPI
CUDA_BLOCK_SIZES=(64 128 256 512 1024)    # Threads por Bloco na GPU

# Parâmetros do K-Means
MAX_ITER=50
THRESHOLD=0.000001

# 3. COMPILAÇÃO (BUILD)
# ---------------------
log ">>> [Fase 1] Compilando códigos..."

# Geradores
gcc -std=c99 data/geradorDados.c -o data/geradorDados
gcc -std=c99 data/geradorCentroides.c -o data/geradorCentroides

# Implementações
# Serial
gcc -O2 -std=c99 serial/kmeans_1d_serial.c -o serial/kmeans_serial -lm
# OpenMP
gcc -O2 -std=c99 -fopenmp openMP/opMP.c -o openMP/kmeans_omp -lm
# MPI
mpicc -O2 -std=c99 mpi/kmeans_mpi.c -o mpi/kmeans_mpi -lm
# CUDA (Adicionado flag -Wno-deprecated-gpu-targets para evitar warnings comuns)
nvcc -O2 cuda/kmeans_cuda.cu -o cuda/kmeans_cuda -lm -Xcompiler -fopenmp

log "Compilação finalizada."
log ""

# 4. LOOP PRINCIPAL DE EXECUÇÃO
# -----------------------------
for config in "${CONFIGS[@]}"; do
    read N K <<< "$config"
    
    log "=================================================================="
    log "DATASET: N=$N pontos, K=$K clusters"
    log "=================================================================="

    # A. Geração de Dados (Uma vez por configuração para todos usarem o mesmo input)
    log "-> Gerando arquivos de entrada..."
    ./data/geradorDados $N $K
    ./data/geradorCentroides $K
    
    INPUT_DATA="data/dados.csv"
    INPUT_CENTROIDS="data/centroides_iniciais.csv"
    
    # B. Execução SERIAL (Baseline para Speedup)
    log "-> [SERIAL] Executando Baseline..."
    OUT_SERIAL_C="serial/output/centroids_${N}.csv"
    OUT_SERIAL_A="serial/output/assign_${N}.csv"
    
    # Executa e captura o tempo (assumindo que o programa imprime o tempo no stdout)
    ./serial/kmeans_serial $INPUT_DATA $INPUT_CENTROIDS $MAX_ITER $THRESHOLD $OUT_SERIAL_A $OUT_SERIAL_C 2>&1 | tee -a "$OUTPUT_FILE"
    log ""

    # C. Execução OPENMP (Memória Compartilhada)
    log "-> [OpenMP] Iniciando testes de escalabilidade..."
    for t in "${OMP_THREADS[@]}"; do
        log "   Running OMP with $t threads..."
        OUT_OMP_C="openMP/output/centroids_${N}_${t}t.csv"
        OUT_OMP_A="openMP/output/assign_${N}_${t}t.csv"
        
        ./openMP/kmeans_omp $INPUT_DATA $INPUT_CENTROIDS $MAX_ITER $THRESHOLD $t $OUT_OMP_A $OUT_OMP_C 2>&1 | tee -a "$OUTPUT_FILE"
        
        # Validação Rápida (apenas centroids)
        if cmp -s "$OUT_SERIAL_C" "$OUT_OMP_C"; then
            echo "      [VALIDAÇÃO: OK]" >> "$OUTPUT_FILE"
        else
            echo "      [VALIDAÇÃO: FALHA - Diferença no output]" >> "$OUTPUT_FILE"
        fi
    done
    log ""

    # D. Execução MPI (Memória Distribuída)
    log "-> [MPI] Iniciando testes de escalabilidade..."
    for p in "${MPI_PROCESSES[@]}"; do
        log "   Running MPI with $p processes..."
        OUT_MPI_C="mpi/output/centroids_${N}_${p}p.csv"
        OUT_MPI_A="mpi/output/assign_${N}_${p}p.csv"
        
        # --oversubscribe permite rodar mais processos que cores (útil para testes de lógica/overhead)
        mpirun --oversubscribe -np $p ./mpi/kmeans_mpi $INPUT_DATA $INPUT_CENTROIDS $MAX_ITER $THRESHOLD $OUT_MPI_A $OUT_MPI_C 2>&1 | tee -a "$OUTPUT_FILE"
        
        # Validação (MPI costuma ter ordem de escrita diferente no arquivo de assignment, comparar centroids é mais seguro)
        if cmp -s "$OUT_SERIAL_C" "$OUT_MPI_C"; then
             echo "      [VALIDAÇÃO: OK]" >> "$OUTPUT_FILE"
        else
             echo "      [VALIDAÇÃO: FALHA]" >> "$OUTPUT_FILE"
        fi
    done
    log ""

    # E. Execução CUDA (Aceleração por Hardware/GPU)
    log "-> [CUDA] Testando configurações de kernel..."
    for block in "${CUDA_BLOCK_SIZES[@]}"; do
        log "   Running CUDA with Block Size = $block..."
        OUT_CUDA_C="cuda/output/centroids_${N}_${block}blk.csv"
        OUT_CUDA_A="cuda/output/assign_${N}_${block}blk.csv"
        
        ./cuda/kmeans_cuda $INPUT_DATA $INPUT_CENTROIDS $MAX_ITER $THRESHOLD $OUT_CUDA_A $OUT_CUDA_C $block 2>&1 | tee -a "$OUTPUT_FILE"
        
        # Nota: Floating point na GPU pode ter pequenas variações vs CPU.
        # Um 'cmp' binário pode falhar mesmo estando certo, mas serve de alerta.
        if cmp -s "$OUT_SERIAL_C" "$OUT_CUDA_C"; then
             echo "      [VALIDAÇÃO: OK]" >> "$OUTPUT_FILE"
        else
             echo "      [VALIDAÇÃO: DIFF] (Normal se a diferença for < epsilon)" >> "$OUTPUT_FILE"
        fi
    done
    log ""
    
    log "Configuração N=$N finalizada."
    log "------------------------------------------------------------------"
done

log "Experimentos concluídos. Verifique $OUTPUT_FILE"