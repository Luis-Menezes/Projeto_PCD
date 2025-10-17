/* kmeans_1d_naive.c
   K-means 1D (C99), implementação "naive":
   - Lê X (N linhas, 1 coluna) e C_init (K linhas, 1 coluna) de CSVs sem cabeçalho.
   - Itera assignment + update até max_iter ou variação relativa do SSE < eps.
   - Salva (opcional) assign (N linhas) e centróides finais (K linhas).

   Compilar: gcc -O2 -std=c99 kmeans_1d_naive.c -o kmeans_1d_naive -lm
   Uso:      ./kmeans_1d_naive dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

/* ---------- util CSV 1D: cada linha tem 1 número ---------- */
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

        /* aceita vírgula/ponto-e-vírgula/espaco/tab, pega o primeiro token numérico */
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

/* ---------- k-means 1D ---------- */
/* AQUI PARALELIZA */
/* assignment: para cada X[i], encontra c com menor (X[i]-C[c])^2 */
static double assignment_step_1d(const double *X, const double *C, int *assign, int N, int K){
    double sse = 0.0;

    // A diretiva "parallel for" divide as iterações do laço 'i' entre os threads.
    // A cláusula "reduction(+:sse)" cria uma cópia local de 'sse' para cada thread.
    // No final, o OpenMP soma todas as cópias locais no 'sse' original de forma segura.
    #pragma omp parallel for reduction(+:sse)
    for(int i=0; i<N; i++){
        int best = -1;
        double bestd = 1e300;
        for(int c=0; c<K; c++){
            double diff = X[i] - C[c];
            double d = diff*diff;
            if(d < bestd){ bestd = d; best = c; }
        }
        assign[i] = best;
        sse += bestd;
    }
    return sse;
}

static double silhouetteSample(const double *X, const double *C, const int *assign, int idx, int N, int K) {
    /* Não paraleliza porque é senão vai ser só serializado */
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

/* update: média dos pontos de cada cluster (1D)
   se cluster vazio, copia X[0] (estratégia naive) */
/* AQUI PARALELIZA DE DUAS FORMAS DIFERENTES */

/* com base nos docs da atividade - OPCAO A */
/*static void update_step_1d_critical(const double *X, double *C, const int *assign, int N, int K){
    double *sum = (double*)calloc((size_t)K, sizeof(double));
    int *cnt = (int*)calloc((size_t)K, sizeof(int));
    if(!sum || !cnt){ fprintf(stderr,"Sem memoria no update\n"); exit(1); }

    #pragma omp parallel for
    for(int i=0; i<N; i++){
        int a = assign[i];
        // A seção crítica garante que apenas um thread por vez execute
        // estas duas linhas, evitando a condição de corrida.
        #pragma omp critical
        {
            cnt[a] += 1;
            sum[a] += X[i];
        }
    }
    
    // O cálculo final dos centróides é feito de forma serial
    for(int c=0; c<K; c++){
        if(cnt[c] > 0) C[c] = sum[c] / (double)cnt[c];
        else           C[c] = X[0];
    }
    free(sum); free(cnt);
}
*/

/* com base nos docs da atividade - OPCAO B */
static void update_step_1d_local_accum(const double *X, double *C, const int *assign, int N, int K){
    // Arrays globais para o resultado final
    double *sum_global = (double*)calloc((size_t)K, sizeof(double));
    int *cnt_global = (int*)calloc((size_t)K, sizeof(int));
    if(!sum_global || !cnt_global){ fprintf(stderr,"Sem memoria no update\n"); exit(1); }

    #pragma omp parallel
    {
        // 1. Cada thread cria seus próprios acumuladores locais.
        double *sum_local = (double*)calloc((size_t)K, sizeof(double));
        int *cnt_local = (int*)calloc((size_t)K, sizeof(int));

        // 2. O laço é dividido. Cada thread atualiza APENAS sua cópia local. Sem concorrência!
        #pragma omp for
        for(int i=0; i<N; i++){
            int a = assign[i];
            cnt_local[a] += 1;
            sum_local[a] += X[i];
        }

        // 3. Após o laço, cada thread adiciona seus resultados locais aos arrays globais.
        // A seção crítica aqui é muito mais rápida, pois é executada apenas uma vez por thread para cada cluster.
        for(int c=0; c<K; c++){
            #pragma omp critical
            {
                sum_global[c] += sum_local[c];
                cnt_global[c] += cnt_local[c];
            }
        }
        
        // Cada thread libera a memória de seus arrays locais.
        free(sum_local);
        free(cnt_local);
    }

    // O restante do código é executado em um único thread.
    for(int c=0; c<K; c++){
        if(cnt_global[c] > 0) C[c] = sum_global[c] / (double)cnt_global[c];
        else                  C[c] = X[0];
    }
    free(sum_global); free(cnt_global);
}

static void kmeans_1d(const double *X, double *C, int *assign,
                      int N, int K, int max_iter, double eps,
                      int *iters_out, double *sse_out)
{
    double prev_sse = 1e300;
    double sse = 0.0;
    int it;
    for(it=0; it<max_iter; it++){
        sse = assignment_step_1d(X, C, assign, N, K);
        /* parada por variação relativa do SSE */
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){ it++; break; }
        //update_step_1d(X, C, assign, N, K);
        //update_step_1d_critical(X, C, assign, N, K); // OPCAO A
        update_step_1d_local_accum(X, C, assign, N, K); // OPCAO B
        prev_sse = sse;
    }
    *iters_out = it;
    *sse_out = sse;
}

/* ---------- main ---------- */
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
    int n_threads = (argc>5)? atoi(argv[5]) : 4;
    const char *outAssign   = (argc>6)? argv[6] : NULL;
    const char *outCentroid = (argc>7)? argv[7] : NULL;
    double silhouette = 0.0;
    if(max_iter <= 0 || eps <= 0.0){
        fprintf(stderr,"Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

        #ifdef _OPENMP
    omp_set_num_threads(n_threads); // Define o número de threads para OpenMP
    printf("OpenMP habilitado com %d threads configuradas.\n", n_threads);
    
    // Verificar quantas threads realmente serão usadas
    int actual_threads;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }
    printf("Threads efetivamente utilizadas: %d\n", actual_threads);
    #else
    printf("OpenMP não habilitado. Compilar com -fopenmp para ativar.\n");
    #endif

    int N=0, K=0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));
    if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); return 1; }

    double t0_kmeans = omp_get_wtime();
    int iters = 0; double sse = 0.0;
    kmeans_1d(X, C, assign, N, K, max_iter, eps, &iters, &sse);
    // kmeans_1d(X, C, assign, N, K, max_iter, eps, &iters, &sse);
    silhouette = calculaSilhouette(X, C, assign, N, K);
    double t1_kmeans = omp_get_wtime();
    double ms =  (double)(t1_kmeans - t0_kmeans) * 1000;

    printf("K-means 1D (naive)\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, ms);
    printf("Tempo medido com omp_get_wtime(): %.6f segundos\n", t1_kmeans - t0_kmeans);
    printf("Coeficiente silhouette médio: %.6f\n", silhouette);

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign); free(X); free(C);
    return 0;
}