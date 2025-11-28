/* kmeans_1d_mpi.c
   K-means 1D (MPI), implementação distribuída modularizada.
   - Rank 0 lê X e C_init.
   - Distribui X entre P processos (MPI_Scatter).
   - Função kmeans_1d_mpi orquestra o laço.
   - Rank 0 recolhe assign final (MPI_Gather) e salva.

   Compilar: mpicc -O2 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm
   Uso:      mpirun -np <P> ./kmeans_1d_mpi dados.csv centroides_iniciais.csv [max_iter] [eps] ...
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/* ---------- Funções Utilitárias (I/O) ---------- */
static int count_rows(const char *path){
    FILE *f = fopen(path, "r");
    if(!f){ return 0; }
    int rows=0; char line[8192];
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0){ return NULL; }
    double *A = (double*)malloc((size_t)R * sizeof(double));
    if(!A){ return NULL; }

    FILE *f = fopen(path, "r");
    char line[8192];
    int r=0;
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(only_ws) continue;
        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if(!tok){ free(A); fclose(f); return NULL; }
        A[r] = atof(tok);
        r++;
        if(r>R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ return; }
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ return; }
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* ---------- Funções Core do K-means (Distribuído) ---------- */

/* 1. Assignment Local: Cada processo calcula para seus dados */
static double assignment_step_local(const double *X_local, const double *C, int *assign_local, int N_local, int K){
    double sse_local = 0.0;
    for(int i=0; i<N_local; i++){
        int best = -1;
        double bestd = 1e300;
        for(int c=0; c<K; c++){
            double diff = X_local[i] - C[c];
            double d = diff*diff;
            if(d < bestd){ bestd = d; best = c; }
        }
        assign_local[i] = best;
        sse_local += bestd;
    }
    return sse_local;
}

/* 2. Update Distribuído: Calcula somas locais e faz a Redução Global */
static void update_step_mpi(const double *X_local, double *C, const int *assign_local, int N_local, int K, double *comm_time_out){
    
    // Alocação de buffers auxiliares
    double *sum_local = (double*)calloc(K, sizeof(double));
    int *cnt_local = (int*)calloc(K, sizeof(int));
    double *sum_global = (double*)malloc(K * sizeof(double));
    int *cnt_global = (int*)malloc(K * sizeof(int));

    // A. Cálculo das somas parciais (LOCAL)
    for(int i=0; i<N_local; i++){
        int a = assign_local[i];
        cnt_local[a] += 1;
        sum_local[a] += X_local[i];
    }

    // B. Comunicação Global (Allreduce)
    double t0 = MPI_Wtime();
    
    // Soma as coordenadas de todos os processos
    MPI_Allreduce(sum_local, sum_global, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // Soma as contagens de todos os processos
    MPI_Allreduce(cnt_local, cnt_global, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    double t1 = MPI_Wtime();
    *comm_time_out += (t1 - t0); // Contabiliza tempo de comunicação

    // C. Atualização dos Centróides (Global) - Todos processos fazem igual
    for(int c=0; c<K; c++){
        if(cnt_global[c] > 0) C[c] = sum_global[c] / (double)cnt_global[c];
        else                  C[c] = C[c]; 
    }

    free(sum_local); free(cnt_local);
    free(sum_global); free(cnt_global);
}

/* 3. Função Principal do Algoritmo (Orquestrador) */
static void kmeans_1d_mpi(const double *X_local, double *C, int *assign_local, 
                          int N_local, int K, int max_iter, double eps,
                          int *iters_out, double *sse_global_out, double *comm_time_out)
{
    double prev_sse = 1e300;
    double sse_global = 0.0;
    *comm_time_out = 0.0;
    int it;

    for(it=0; it<max_iter; it++){
        
        // Passo 1: Assignment (Local)
        double sse_local = assignment_step_local(X_local, C, assign_local, N_local, K);

        // Passo 2: Redução do SSE (Comunicação)
        double t0 = MPI_Wtime();
        MPI_Allreduce(&sse_local, &sse_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        *comm_time_out += (t1 - t0);

        // Critério de parada
        double rel = fabs(sse_global - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){ it++; break; } // Importante: break antes do update se convergiu
        prev_sse = sse_global;

        // Passo 3: Update (Distribuído)
        update_step_mpi(X_local, C, assign_local, N_local, K, comm_time_out);
    }
    
    *iters_out = it;
    *sse_global_out = sse_global;
}

/* ---------- MAIN ---------- */
int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 0, K = 0;
    double *X_full = NULL;
    double *C = NULL; 
    
    // --- 1. Rank 0 lê os arquivos ---
    if(rank == 0){
        if(argc < 3){
            fprintf(stderr, "Uso: mpirun -np <P> %s dados.csv centroides.csv ...\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        X_full = read_csv_1col(argv[1], &N);
        C = read_csv_1col(argv[2], &K);
        if(!X_full || !C) MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // --- 2. Broadcast de metadados ---
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int N_local = N / size; // Divisão simplificada (truncada)
    
    double *X_local = (double*)malloc(N_local * sizeof(double));
    int *assign_local = (int*)malloc(N_local * sizeof(int));
    if(rank != 0) C = (double*)malloc(K * sizeof(double));

    // --- 3. Distribuição dos Dados (Scatter) ---
    MPI_Scatter(X_full, N_local, MPI_DOUBLE, 
                X_local, N_local, MPI_DOUBLE, 
                0, MPI_COMM_WORLD);

    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD); // C inicial para todos

    // Parâmetros
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps   = (argc>4)? atof(argv[4]) : 1e-4;

    // --- 4. Execução do K-means ---
    double t_start = MPI_Wtime();
    int iters = 0;
    double sse_final = 0.0;
    double comm_time = 0.0;

    kmeans_1d_mpi(X_local, C, assign_local, N_local, K, max_iter, eps, 
                  &iters, &sse_final, &comm_time);

    double t_end = MPI_Wtime();

    // --- 5. Recolher Resultados (Gather) ---
    int *assign_full = NULL;
    if(rank == 0) assign_full = (int*)malloc(N * sizeof(int));
    
    MPI_Gather(assign_local, N_local, MPI_INT, 
               assign_full, N_local, MPI_INT, 
               0, MPI_COMM_WORLD);

    // --- 6. Relatório ---
    if(rank == 0){
        printf("K-means 1D (MPI Distribuído)\n");
        printf("N=%d K=%d P=%d processos\n", N, K, size);
        printf("Iterações: %d | SSE final: %.6f\n", iters, sse_final);
        double total_time = t_end - t_start;
        printf("Tempo Total: %.4f s\n", total_time);
        printf("Tempo Comunicação: %.4f s (%.1f%%)\n", 
               comm_time, (comm_time/total_time)*100.0);

        // Salvar
        const char *outAssign = (argc>5)? argv[5] : NULL;
        const char *outCentroid = (argc>6)? argv[6] : NULL;
        write_assign_csv(outAssign, assign_full, N);
        write_centroids_csv(outCentroid, C, K);

        free(X_full);
        free(assign_full);
    }

    // Limpeza
    free(X_local);
    free(assign_local);
    free(C);

    MPI_Finalize();
    return 0;
}