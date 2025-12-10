/* kmeans_1d_cuda.cu
   K-means 1D (C99/CUDA), implementação "naive" com Opção A (Update no Host)
   - Lê X (N linhas, 1 coluna) e C_init (K linhas, 1 coluna) de CSVs sem cabeçalho.
   - Itera assignment (GPU) + update (CPU) até max_iter ou variação relativa do SSE < eps.
   - Salva (opcional) assign (N linhas) e centróides finais (K linhas).

   Compilar: nvcc -O2 -std=c99 kmeans_1d_cuda.cu -o kmeans_1d_cuda -lm -Xcompiler -fopenmp
   Uso:      ./kmeans_1d_cuda dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h> // Adicionado para CUDA
// #include <device_launch_parameters.h>

/* ---------- Funções Utilitárias (CSV) - Inalteradas ---------- */
static int count_rows(const char *path){
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); exit(1); }
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
    if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
    double *A = (double*)malloc((size_t)R * sizeof(double));
    if(!A){ fprintf(stderr,"Sem memoria para %d linhas\n", R); exit(1); }

    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); free(A); exit(1); }

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
        if(!tok){ fprintf(stderr,"Linha %d sem valor em %s\n", r+1, path); free(A); fclose(f); exit(1); }
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
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* ---------- KERNEL CUDA (DEVICE) ---------- */
/* Kernel de Assignment: 1 thread por ponto i */
__global__ void assignment_kernel(const double *X, const double *C, int *assign,
                                  double *sse_per_point, int N, int K) 
{
    // 1. Descobrir qual ponto (i) este thread deve processar
    int i = blockIdx.x * blockDim.x + threadIdx.x;
 
    // 2. Garantir que o thread não está fora dos limites
    if (i < N) {
        // 3. Lógica do K-means (copiada da função serial)
        int best_c = -1;
        double best_dist = 1e300;
 
        // Cada thread varre K centróides
        for (int c = 0; c < K; c++) {
            double diff = X[i] - C[c];
            double dist = diff * diff; 
            if (dist < best_dist) {
                best_dist = dist;
                best_c = c;
            }
        }
 
        // 4. Escrever resultados na memória global da GPU
        assign[i] = best_c;     
        sse_per_point[i] = best_dist; // Para reduzir o SSE no host 
    }
}


/* ---------- Funções de Update (HOST) ---------- */
/* Opção A: update no host (CPU), usando uma versão serial simples */
static void update_step_1d_serial(const double *X, double *C, const int *assign, int N, int K){
    double *sum = (double*)calloc((size_t)K, sizeof(double));
    int *cnt = (int*)calloc((size_t)K, sizeof(int));
    if(!sum || !cnt){ fprintf(stderr,"Sem memoria no update\n"); exit(1); }

    for(int i=0; i<N; i++){
        int a = assign[i];
        cnt[a] += 1;
        sum[a] += X[i];
    }
    for(int c=0; c<K; c++){
        if(cnt[c] > 0) C[c] = sum[c] / (double)cnt[c];
        else           C[c] = X[0]; /* simples: cluster vazio recebe o primeiro ponto */
    }
    free(sum); free(cnt);
}


static double silhouetteSample(const double *X, const double *C, const int *assign, int idx, int N, int K) {
    /* Não paraleliza porque senão vai ser só serializado -> aumenta muito a granularidade (invés de 32 threads vai gerar 32**2 threads)*/
    int cluster = assign[idx];
    double a = 0.0; // média da distância intra-cluster
    double b = 1e300; // mínima média da distância ao outro cluster

    int count_a = 0;
    for (int j = 0; j < N; j++) {
        if (j == idx) continue; // não conta a si mesmo
        if (assign[j] == cluster) {
            a += fabs(X[idx] - X[j]);
            count_a++;
        }
    }
    // Calcula a média intra-cluster
    if (count_a > 0) a /= count_a;
    else return 0.0; // ponto isolado

    // Calcula a menor distância média inter-cluster (b)
    for (int c = 0; c < K; c++) {
        if (c == cluster) continue;
        
        double dist_sum = 0.0;
        int count_b = 0;
        for (int j = 0; j < N; j++) {
            if (assign[j] == c) {
                dist_sum += fabs(X[idx] - X[j]);
                count_b++;
            }
        }
        
        if (count_b > 0) {
            double avg_dist = dist_sum / count_b;
            if (avg_dist < b) b = avg_dist;
        }
    }

    if(b== 1e300) return 0.0; // não há outro cluster
    if (a==b) return 0.0;
    else return (b - a) / fmax(a, b);
}
/* Implementado com base na implementação do scikit-learn: 
https://github.com/scikit-learn/scikit-learn/blob/c60dae20604f8b9e585fc18a8fa0e0fb50712179/sklearn/metrics/cluster/_unsupervised.py#L51 */
static double calculaSilhouette(const double *X, const double *C, const int *assign, int N, int K){
    double silhouette_sum = 0.0;

    #pragma omp parallel for reduction(+:silhouette_sum)
    for(int i=0; i<N; i++){
        silhouette_sum += silhouetteSample(X, C, assign, i, N, K);
    }
    return silhouette_sum / N;
}


/* ---------- k-means 1D (HOST) - Orquestrador ---------- */
static void kmeans_1d(
    // Ponteiros do Host
    const double *X_h, double *C_h, int *assign_h, double *sse_per_point_h,
    // Ponteiros do Device
    const double *X_d, double *C_d, int *assign_d, double *sse_per_point_d,
    // Parâmetros de controle
    int N, int K, int max_iter, double eps, int threadsPerBlock,
    // Saídas
    int *iters_out, double *sse_out,
    // Saídas de tempo 
    float *h2d_ms_out, float *kernel_ms_out, float *d2h_ms_out)
{
    // Eventos CUDA para medição de tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    *h2d_ms_out = 0.0;
    *kernel_ms_out = 0.0;
    *d2h_ms_out = 0.0;

    // Configuração do Kernel (1 thread por ponto)
     // Valor comum para começar 
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    double prev_sse = 1e300;
    double sse = 0.0;
    int it;

    for(it=0; it<max_iter; it++){
        
        // 1. Copiar centróides C para a GPU (H2D)
        cudaEventRecord(start);
        cudaMemcpy(C_d, C_h, K * sizeof(double), cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        *h2d_ms_out += ms;

        // 2. Chamar o Kernel (Execução na GPU)
        cudaEventRecord(start);
        assignment_kernel<<<numBlocks, threadsPerBlock>>>(
            X_d, C_d, assign_d, sse_per_point_d, N, K
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        *kernel_ms_out += ms;

        // 3. Copiar resultados (assign e sse_per_point) de volta para a CPU (D2H) 
        cudaEventRecord(start);
        cudaMemcpy(assign_h, assign_d, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(sse_per_point_h, sse_per_point_d, N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        *d2h_ms_out += ms;

        // 4. Reduzir o SSE no Host (CPU)
        sse = 0.0;
        for(int i = 0; i < N; i++) {
            sse += sse_per_point_h[i];
        }
        
        // 5. Parada por variação relativa do SSE
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){ it++; break; }

        // 6. Opção A: Update no Host (CPU) 
        update_step_1d_serial(X_h, C_h, assign_h, N, K);
        
        prev_sse = sse;
    }
    *iters_out = it;
    *sse_out = sse;

    // Limpar eventos
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/* ---------- main (HOST) ---------- */
int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]\n", argv[0]);
        printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps   = (argc>4)? atof(argv[4]) : 1e-4;
    const char *outAssign   = (argc>5)? argv[5] : NULL;
    const char *outCentroid = (argc>6)? argv[6] : NULL;
    int threadsPerBlock = (argc>7)? atoi(argv[7]) : 256;

    if(max_iter <= 0 || eps <= 0.0){
        fprintf(stderr,"Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    // --- 1. Alocação de Memória no Host (CPU) ---
    int N=0, K=0;
    double *X_h = read_csv_1col(pathX, &N);
    double *C_h = read_csv_1col(pathC, &K);
    int *assign_h = (int*)malloc((size_t)N * sizeof(int));
    double *sse_per_point_h = (double*)malloc((size_t)N * sizeof(double));
    
    // NOVO: Buffer Host para scores do Silhouette
    double *silhouette_scores_h = (double*)malloc((size_t)N * sizeof(double));

    if(!assign_h || !sse_per_point_h || !silhouette_scores_h){ 
        fprintf(stderr,"Sem memoria host\n"); exit(1); 
    }

    // --- 2. Alocação de Memória no Device (GPU) ---
    double *X_d, *C_d, *sse_per_point_d;
    int *assign_d;
    // Buffer Device para scores do Silhouette
    double *silhouette_scores_d; 
    
    cudaMalloc((void**)&X_d, (size_t)N * sizeof(double));
    cudaMalloc((void**)&C_d, (size_t)K * sizeof(double));
    cudaMalloc((void**)&assign_d, (size_t)N * sizeof(int));
    cudaMalloc((void**)&sse_per_point_d, (size_t)N * sizeof(double));
    cudaMalloc((void**)&silhouette_scores_d, (size_t)N * sizeof(double)); // Aloca

    // --- 3. Cópia Inicial (H2D) ---
    cudaMemcpy(X_d, X_h, (size_t)N * sizeof(double), cudaMemcpyHostToDevice);
    
    // --- 4. Execução e Medição do K-means ---    
    int iters = 0; double sse = 0.0;
    float h2d_ms = 0.0, kernel_ms = 0.0, d2h_ms = 0.0, total_kmeans_ms = 0.0;
    
    cudaEvent_t start_total, stop_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    
    cudaEventRecord(start_total);
    kmeans_1d(X_h, C_h, assign_h, sse_per_point_h,
              X_d, C_d, assign_d, sse_per_point_d,
              N, K, max_iter, eps, threadsPerBlock, &iters, &sse,  
              &h2d_ms, &kernel_ms, &d2h_ms);
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);
    cudaEventElapsedTime(&total_kmeans_ms, start_total, stop_total);

    // --- 5. Execução e Medição do Silhouette (na GPU) ---
    // O kmeans_1d deixa os assignments finais em assign_h.
    // Precisamos copiar a versão final de assign_h para assign_d
    // para que o silhouette_kernel use os dados corretos.
    cudaMemcpy(assign_d, assign_h, (size_t)N * sizeof(int), cudaMemcpyHostToDevice);
    
    float ms_silhouette = 0.0;
    double silhouette = calculaSilhouette(X_h, C_h, assign_h, N, K);


    // --- 6. Impressão de Resultados ---
    printf("K-means 1D (CUDA - Opção A)\n");
    printf("N=%d K=%d max_iter=%d eps=%g threadsPerBlock=%d\n", N, K, max_iter, eps, threadsPerBlock);
    printf("Iterações: %d | SSE final: %.6f\n", iters, sse);
    printf("--- Tempos K-means (ms) ---\n");
    printf("  Tempo H2D (cópias C): %.1f ms\n", h2d_ms);
    printf("  Tempo Kernel (GPU):   %.1f ms\n", kernel_ms);
    printf("  Tempo D2H (cópias A): %.1f ms\n", d2h_ms);
    printf("  Tempo Total K-means:  %.1f ms\n", total_kmeans_ms);
    printf("--- Tempos Outros (ms) ---\n");
    printf("  Tempo Silhouette (GPU): %.1f ms\n", ms_silhouette); // Agora medido pela GPU
    printf("Coeficiente silhouette médio: %.6f\n", silhouette);

    // --- 7. Salvar Saídas ---
    write_assign_csv(outAssign, assign_h, N);
    write_centroids_csv(outCentroid, C_h, K);

    // --- 8. Limpeza de Memória ---
    free(X_h);
    free(C_h);
    free(assign_h);
    free(sse_per_point_h);
    free(silhouette_scores_h); // Limpa novo buffer
    
    cudaFree(X_d);
    cudaFree(C_d);
    cudaFree(assign_d);
    cudaFree(sse_per_point_d);
    cudaFree(silhouette_scores_d); // Limpa novo buffer
    
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);

    return 0;
}