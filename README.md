# Pipeline Modularizado de Classificação — Churn

## Integrantes e responsabilidades

- Pedro Henrique Nunes Zanette — Engenharia de Dados
  Responsável por `data_processing.py`.

- Guilherme Santos Oliveira — Ciência de Dados
  Responsável por `model_training.py`.

- Pedro Henrique Nunes Zanette + Guilherme Santos Oliveira — MLOps & Integração
  Responsáveis por `main.py`.

## Objetivo

Identificar clientes inativos com risco de churn a partir do histórico de
transações, usando um pipeline modularizado em Python (Engenharia de Dados +
Ciência de Dados + Integração/MLOps).

## Estrutura

```
data/raw_transactions.csv   # dataset bruto
data_processing.py          # leitura, pd.pivot_table, geração da flag churn
model_training.py           # split treino/teste, treino e métricas do modelo
main.py                     # orquestração do pipeline e inferência
```

## Regra de churn

Por cliente, soma-se a frequência de compras em todas as categorias
(`frequencia_total`). Se `frequencia_total < 8`, o cliente é `churn = 1`;
caso contrário, `churn = 0`.

## Modelo escolhido

`DecisionTreeClassifier(max_depth=3, random_state=42)`. Como o churn foi
definido por uma regra de frequência, uma árvore de decisão rasa é simples,
interpretável e adequada para demonstrar esse limite de decisão.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução

```bash
python main.py
```
