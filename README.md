# Monitoramento de modelos de _machine learning_
API para avaliação de aderência e performance

Implementação da API e instruções de execução: [App](./app/)

Notebook com os testes das rotas: [main.ipynb](./main.ipynb)

## Descrição do projeto
Uma ferramenta para monitoramento de um modelo de concessão de crédito contendo os seguintes *endpoints*

### Performance
Recebe:
* Uma lista de registros

Retorna:
* A quantidade de registros para cada mês presente na lista dada;
* Performance do modelo pré-treinado neste conjunto de registros, indicada pelo valor da área sob a curva ROC.

### Aderência
Recebe:
* Um caminho para um arquivo dataset armazenado localmente (no servidor)

Retorna:
* O quanto a distribuição de scores da base está distante, ou diferente, da distribuição vista na base de Teste da modelagem utilizando o teste estatístico de Kolmogorov-Smirnov (KS).
* P-Value do teste estatístico

### Outras *features*
* Docker: o projeto tem a capacidade de ser rodado dentro de um *docker container* através do arquivo [Dockerfile](./app/Dockerfile)
* XGBoost: um modelo XGBoost foi treinado para performar a tarefa de análise de crédito com objetivo de melhorar a performance sobre o modelo dado previamente. Detalhes no arquivo [./xgboost.ipynb](./xgboost.ipynb)
