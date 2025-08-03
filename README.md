# Análise de Churn da Telecom X: Do Dado à Predição

## 1. Sobre o Projeto

Este projeto foi desenvolvido como parte do **Challenge de Machine Learning da Alura**, em parceria com a **Oracle (ONE)**. O objetivo é construir um pipeline completo de Machine Learning para prever a evasão de clientes (churn) da empresa fictícia **Telecom X**. A solução abrange desde a preparação dos dados até o treinamento, avaliação e implantação de modelos preditivos, com foco em gerar insights estratégicos para a tomada de decisão.

O modelo final, baseado em **Random Forest**, alcançou uma **acurácia de aproximadamente 79%**, demonstrando ser uma ferramenta eficaz para identificar clientes com alto risco de cancelamento.

## 2. Estrutura do Repositório

O projeto está organizado em uma sequência de scripts Python que representam cada etapa do pipeline de Machine Learning:

```
/TelecomX_BR_2
│
├── 01_data_preparation.py      # Carrega, limpa e transforma os dados brutos (JSON -> CSV).
├── 02_feature_engineering.py   # Aplica One-Hot Encoding e escalonamento nas features.
├── 03_model_training.py        # Treina e avalia os modelos (Regressão Logística e Random Forest).
├── 04_ml_pipeline.py           # Consolida o pré-processamento e o modelo em um pipeline e o salva.
│
├── random_forest_pipeline.pkl  # Pipeline final treinado e salvo, pronto para predições.
│
├── data/
│   ├── TelecomX_Data.json      # Dados brutos originais.
│   └── telecom_data_tratado.csv # Dados limpos e pré-processados.
│
├── notebooks/
│   └── TelecomX_BR_2_Completo.ipynb # Notebook com a análise exploratória e desenvolvimento.
│
└── README.md                   # Este arquivo.
```

## 3. Metodologia

O fluxo de trabalho seguiu as melhores práticas de ciência de dados, dividido nas seguintes etapas:

1.  **Preparação dos Dados**: Os dados foram carregados de um arquivo JSON, normalizados para um formato tabular, e submetidos a um processo de limpeza que incluiu:
    *   Tratamento de valores numéricos incorretos.
    *   Remoção de registros com dados de churn ausentes.
    *   Conversão da variável alvo (`Churn`) para formato binário (0 ou 1).
    *   Padronização de valores categóricos inconsistentes.

2.  **Engenharia de Features**: As variáveis foram preparadas para a modelagem por meio de:
    *   **One-Hot Encoding**: Transformação de variáveis categóricas em formato numérico para que o modelo possa interpretá-las.
    *   **Escalonamento**: Normalização das variáveis numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`) com `StandardScaler` para que tivessem a mesma escala de magnitude.

3.  **Treinamento e Avaliação de Modelos**: Foram implementados e comparados dois algoritmos de classificação:
    *   **Regressão Logística**: Um modelo linear simples e interpretável, utilizado como baseline.
    *   **Random Forest**: Um modelo de ensemble mais robusto, que apresentou melhor performance e permitiu a análise de importância das variáveis.

4.  **Criação do Pipeline**: A etapa final consolidou o pré-processamento e o modelo Random Forest em um único objeto `Pipeline` do Scikit-learn. Isso simplifica o fluxo de trabalho e garante que os mesmos passos de transformação sejam aplicados de forma consistente em novos dados.

## 4. Principais Fatores de Churn Identificados

A análise de importância de features do modelo Random Forest revelou os principais fatores que influenciam a decisão de um cliente cancelar o serviço. Em ordem de importância, são eles:

1.  **Tipo de Contrato (`Contract`)**: Clientes com contratos mensais (`Month-to-month`) são, de longe, os mais propensos a sair. A ausência de uma multa contratual representa uma baixa barreira para o cancelamento.
2.  **Fidelidade (`tenure`)**: Clientes com poucos meses de serviço estão em um "período crítico". A probabilidade de churn diminui drasticamente à medida que o cliente permanece mais tempo na empresa.
3.  **Serviço de Internet (`InternetService_Fiber optic`)**: Clientes com serviço de Fibra Ótica apresentam uma taxa de churn mais elevada. Isso pode indicar problemas na oferta, como preço percebido como alto, instabilidade técnica ou concorrência agressiva.
4.  **Forma de Pagamento (`PaymentMethod_Electronic check`)**: Clientes que utilizam boleto eletrônico, que exige uma ação manual mensal, tendem a cancelar mais do que aqueles com métodos de pagamento automáticos.

## 5. Como Executar o Projeto

Para replicar a análise e treinar o modelo, siga os passos abaixo.

### Pré-requisitos

- Python 3.x
- Bibliotecas: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`

Você pode instalar as dependências com:
```bash
pip install pandas scikit-learn matplotlib seaborn joblib
```

### Executando o Pipeline

Os scripts foram projetados para serem executados em sequência. Abra um terminal no diretório do projeto e execute os comandos na ordem:

```bash
# 1. Prepara os dados (JSON -> CSV)
python 01_data_preparation.py

# 2. Cria as features para a modelagem
python 02_feature_engineering.py

# 3. Treina e avalia os modelos (opcional, para análise)
python 03_model_training.py

# 4. Cria e salva o pipeline final de ML
python 04_ml_pipeline.py
```

Ao final da execução, o arquivo `random_forest_pipeline.pkl` será gerado, contendo o modelo treinado e pronto para ser usado em predições.

## 6. Recomendações Estratégicas

Com base nos resultados, as seguintes ações são recomendadas para a Telecom X:

- **Ação para Contratos Mensais (Maior Impacto)**: Criar campanhas proativas para migrar clientes de planos mensais para anuais, oferecendo descontos ou benefícios.
- **Ação para Novos Clientes (Retenção Inicial)**: Implementar um programa de "Onboarding de Sucesso" nos primeiros 3 meses para garantir uma boa experiência inicial.
- **Ação para Clientes de Fibra Ótica (Investigação Crítica)**: Realizar uma análise aprofundada para entender a causa do churn neste segmento (preço, qualidade, suporte ou concorrência).
- **Ação para Forma de Pagamento**: Incentivar a adesão ao débito automático, oferecendo pequenos descontos na fatura.

## 7. Conclusão

Este projeto demonstra o poder do Machine Learning não apenas para prever resultados, but para fornecer um mapa claro de onde os esforços de negócio devem ser concentrados. A implementação das estratégias recomendadas tem o potencial de reduzir significativamente a evasão de clientes e fortalecer o relacionamento da Telecom X com sua base de clientes.
