# 📊 Análise Preditiva de Churn para a Telecom X

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/status-Concluído-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 📚 Sumário

- [Visão Geral do Projeto](#visão-geral-do-projeto)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Metodologia Aplicada](#metodologia-aplicada)
- [Fatores-Chave de Churn](#fatores-chave-de-churn)
- [Como Executar o Notebook](#como-executar-o-notebook)
- [Recomendações Estratégicas](#recomendações-estratégicas)
- [Conclusão](#conclusão)

---

## 🚀 Visão Geral do Projeto

Este repositório contém a solução completa para o **Challenge de Machine Learning da Alura em parceria com a Oracle (ONE)**. O desafio consistiu em desenvolver um sistema de previsão de evasão de clientes (churn) para a **Telecom X**, uma empresa fictícia do setor de telecomunicações.

O projeto abrange todo o ciclo de vida de um projeto de Machine Learning, desde a análise e preparação dos dados até o treinamento, avaliação de modelos e a criação de um pipeline de inferência. O resultado é um modelo preditivo robusto, com **acurácia de 79%**, capaz de identificar clientes com alta probabilidade de cancelamento, fornecendo à empresa uma ferramenta estratégica para campanhas de retenção.

---

## 🗂️ Estrutura do Repositório

```text
/TelecomX_BR_2
│
├── data/
│   ├── TelecomX_Data.json        # Dados brutos fornecidos no desafio
│   └── telecom_data_tratado.csv  # Dados processados
│
├── notebooks/
│   └── TelecomX_BR_2_Completo.ipynb # Notebook original com EDA e desenvolvimento
│
└── README.md                     # Documentação do projeto
```

---

## 🧠 Metodologia Aplicada

O desenvolvimento seguiu uma abordagem estruturada, dividida em quatro etapas principais:

1. **Preparação dos Dados**  
   - Carga dos dados em JSON  
   - Limpeza de inconsistências, correção de tipos e remoção de registros incompletos

2. **Engenharia de Features**  
   - Conversão de variáveis categóricas via **One-Hot Encoding**
   - Normalização de variáveis numéricas com **StandardScaler**

3. **Treinamento e Avaliação**  
   - Modelos: **Regressão Logística** (baseline) e **Random Forest** (final)
   - Avaliação: acurácia, precision, recall, F1-score e matriz de confusão

4. **Construção do Pipeline**  
   - Pipeline do Scikit-learn integrando pré-processamento e modelo
   - Pronto para uso em produção e predições em tempo real

---

## 🔑 Fatores-Chave de Churn

A análise de importância de features revelou os principais indicadores de churn:

- **Tipo de Contrato (`Contract`)**: Contratos mensais são o principal fator de risco.
- **Tempo de Contrato (`tenure`)**: Clientes com baixo tempo de casa têm maior probabilidade de churn.
- **Serviço de Internet (`InternetService_Fiber optic`)**: Fibra ótica está associada a churn mais alto, sugerindo possíveis problemas de expectativa vs. realidade.

---

## 💻 Como Executar o Notebook

### ⚙️ Pré-requisitos

- Python 3.8 ou superior
- Jupyter Notebook ou JupyterLab
- Instale as dependências:
  
  ```bash
  pip install pandas scikit-learn matplotlib seaborn joblib
  ```

### ▶️ Execução

1. Abra o arquivo [`notebooks/TelecomX_BR_2_Completo.ipynb`](notebooks/TelecomX_BR_2_Completo.ipynb) no Jupyter Notebook ou JupyterLab.
2. Execute as células sequencialmente para reproduzir toda a análise, desde a preparação dos dados até a geração dos insights e recomendações.

> 💡 **Dica:** Você pode executar o notebook online pelo [Google Colab](https://colab.research.google.com/) arrastando o arquivo `.ipynb` para a plataforma.

---

## 💡 Recomendações Estratégicas

Os insights do modelo permitem à Telecom X adotar uma postura proativa na retenção de clientes:

- 🎯 **Foco em Contratos de Longo Prazo**: Incentivar a migração de clientes de planos mensais para contratos anuais ou bianuais, oferecendo benefícios claros.
- 🤝 **Programa de Onboarding para Novos Clientes**: Criar uma jornada de boas-vindas para garantir uma experiência positiva nos primeiros meses.
- 🔍 **Análise da Oferta de Fibra Ótica**: Investigar as causas da alta taxa de churn entre clientes de fibra, avaliando preços, qualidade e suporte.

---

## 🏁 Conclusão

Este projeto entrega não apenas um modelo de Machine Learning, mas uma solução de ponta a ponta que traduz dados brutos em inteligência de negócio.  
A capacidade de prever o churn com alta precisão permite que a Telecom X direcione seus recursos de forma mais eficiente, maximize a retenção de clientes e aumente sua receita e competitividade no mercado.

---

## 📬 Contato

Dúvidas ou sugestões? Abra uma [issue](https://github.com/seuusuario/seurepositorio/issues) ou entre em contato!

---

<p align="center">
  <img src="https://img.shields.io/badge/feito%20com-❤%20por%20ONE%20e%20Alura-blue" alt="Feito com amor por ONE e Alura"/>
</p>