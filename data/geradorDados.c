#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Gerador de dados para utilizacao na verificacao e validacao do K-means */
int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s <num_pontos_N> <num_clusters_K>\n", argv[0]);
        printf("Tamanho escolhido para o conjunto pequeno! %s 10000 4\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int K = atoi(argv[2]);

    if(N <= 0 || K <= 0){
        printf("Erro: N e K devem ser maiores que zero!\n");
        return 1;
    }

    if (N % K != 0){
        printf("Erro: Para garantir balanceamento, N deve ser divisivel por K!\n");
        return 1;
    }

    /* Centros com capacidade de ate 16 clusters - espacados */
    double centros[] = {
        10.0, 50.0, 150.0, 200.0, // K = 4
        300.0, 400.0, 500.0, 600.0, // K = 8
        750.0, 900.0, 1050.0, 1200.0, // K = 12
        1400.0, 1600.0, 1800.0, 2000.0 // K = 16
    };

    /* Pega num maximo de clusters que o array suporta */
    size_t max_k = sizeof(centros) / sizeof(double);
    if(K > max_k){
        printf("Erro: Foi escolhido K = %d, mas o maximo suportado eh %zu!\n", K, max_k);
        return 1;
    }

    /* Controla o espalhamento dos pontos - valor pequeno cria clusters densos */
    double spread = 5.0;

    /* A sememente fixa garante que os dados gerados sejam sempre os mesmos - reprodutibilidade dos testes */
    srand(42);

    FILE *arquivoD = fopen("dados.csv", "w");
    if(arquivoD == NULL){
        perror("Erro ao criar o arquivo de dados!\n");
        return 1;
    }

    /* Gera exatamente N/K pontos para cada um dos K clusters */
    int pontos_por_cluster = N / K;
    for(int i=0; i<K; i++){
        for(int j=0; j<pontos_por_cluster; j++){
            /* cria um num aleatorio entre -spread e +spread - ruido aleatorio para somar ao centro */
            double ruido = ((double)rand() / RAND_MAX - 0.5) * 2 * spread;
            double ponto = centros[i] + ruido;
            fprintf(arquivoD, "%.4f\n", ponto);
        }
    }
    fclose(arquivoD);
    printf("Arquivo dos dados com %d pontos e %d clusters gerado com sucesso!\n", N, K);
    return 0;
}