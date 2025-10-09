# Implementação Serial

A implementação serial do algoritmo K-Means Naive lê pontos e centróides iniciais de CSVs de uma coluna, itera os passos de *assignment* (atribuição pelo menor quadrado da distância) e *update* (média dos pontos por cluster) até atingir um número máximo de iterações ou uma variação relativa do SSE abaixo de *eps*. O código usa arrays C, X e assign simples em memória, trata clusters vazios copiando o primeiro ponto (estratégia simples) e mede tempo de execução, o que a torna um bom baseline para testes e validação. Por ser *single‑threaded* e de complexidade $O(N·K·iter)$, não é ideal para conjuntos muito grandes, mas é clara, fácil de entender e serve como ponto de partida para as otimizações realizadas nesse projeto.

## Como compilar: