# Geração dos dados

Os dados consistirão num CSV com 1 coluna. O script para gerar os dados está nessa pasta como ```gen_data.c``` e irá produzir um arquivo CSV com uma coluna e $N$ linhas e deve ser compilado da seguinte forma:

``` bash
gcc gen_data.c -o gen_data
./gen_data
```

Para geração dos centróides iniciais, iremos utilizar o método de inicialização aleatória dos clusters, no script ``` get_init_centroids.c ``` que gerará um CSV com uma coluna e $K$ linhas, e deve ser compilado da seguinte forma:

```bash
gcc get_init_centroids.c -o get_init_centroids
./get_init_centroids 
```

Para compilação dos arquivos de geração dos dados, executar:
Vale ressaltar que os valores das linhas 22 e 23 são alterados conforme o número de dados que se deseja gerar
```bash
gcc -std=c99 geradorDados.c -o geradorDados
gcc -std=c99 geradorCentroides.c -o geradorCentroides
./geradorDados 100000 8
./geradorCentroides 8 
```