#include <stdio.h>
#include <stdlib.h>

/* Gerador de centroides para utilizacao na verificacao e validacao do K-means */
int main(int argc, char **argv){
    if(argc < 2){
        printf("Uso: %s <num_clusters_K>\n", argv[0]);
        printf("Escolha outro valor para K!");
        return 1;
    }

    int K = atoi(argv[1]);
    if(K<=0){
        printf("Erro: K deve ser maior que zero!\n");
        return 1;
    }

    /* Usando os mesmos centros do geradorDados.c */
    double centros[] = {
        10.0, 50.0, 150.0, 200.0, // K = 4
        300.0, 400.0, 500.0, 600.0, // K = 8
        750.0, 900.0, 1050.0, 1200.0, // K = 12
        1400.0, 1600.0, 1800.0, 2000.0 // K = 16
    };

    size_t max_k = sizeof(centros) / sizeof(double);
    if(K>max_k){
        printf("Erro: Foi escolhido K = %d, mas o maximo suportado eh %zu!\n", K, max_k);
        return 1;
    }

    /* Pequeno desvio para os centroides iniciais para nao comecarem do mesmo lugar */
    double desvio = 2.0;

    FILE *arquivoC = fopen("data/centroides_iniciais.csv", "w");
    if(arquivoC == NULL){
        perror("Erro ao criar o arquivo de centroides!\n");
        return 1;
    }

    for(int i=0; i<K; i++){
        fprintf(arquivoC, "%.4f\n", centros[i] + desvio);
    }

    fclose(arquivoC);
    printf("Arquivo dos centroides com K=%d centroides gerado com sucesso!\n", K);
    return 0;
}