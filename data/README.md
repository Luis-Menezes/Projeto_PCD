# Geração dos dados

Os dados consistirão num CSV com 1 coluna. O script para gerar os dados está nessa pasta como ```geradorDados.c``` e irá produzir um arquivo CSV com uma coluna e $N$ linhas.

Para geração dos centróides iniciais, iremos utilizar o método de inicialização aleatória dos clusters, no script ``` geradorCentroides.c ``` que gerará um CSV com uma coluna e $K$ linhas.

Para compilação dos arquivos de geração dos dados, executar:
Vale ressaltar que os valores das  são alterados conforme o número de dados que se deseja gerar
```bash
gcc -std=c99 geradorDados.c -o geradorDados
gcc -std=c99 geradorCentroides.c -o geradorCentroides
./geradorDados <N_dados> <K_clusters>
./geradorCentroides <K_clusters> 
```