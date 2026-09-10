"""Engenharia de Dados: leitura, pivot_table e geração da flag de churn."""

import pandas as pd

COLUNAS_ESPERADAS = ["id_transacao", "id_cliente", "categoria", "valor"]
FREQUENCIA_MINIMA_ATIVO = 8


def carregar_dados(caminho_csv="data/raw_transactions.csv"):
    df = pd.read_csv(caminho_csv)

    colunas_faltantes = set(COLUNAS_ESPERADAS) - set(df.columns)
    if colunas_faltantes:
        raise ValueError(f"Colunas faltando no CSV: {colunas_faltantes}")

    return df


def processar_dados(df):
    tabela = pd.pivot_table(
        df,
        index="id_cliente",
        columns="categoria",
        values="valor",
        aggfunc=["sum", "count"],
        fill_value=0,
    )

    tabela.columns = [
        f"gasto_{categoria.lower()}" if agregacao == "sum" else f"freq_{categoria.lower()}"
        for agregacao, categoria in tabela.columns
    ]
    tabela = tabela.reset_index()

    colunas_freq = [c for c in tabela.columns if c.startswith("freq_")]
    colunas_gasto = [c for c in tabela.columns if c.startswith("gasto_")]

    tabela["frequencia_total"] = tabela[colunas_freq].sum(axis=1)
    tabela["gasto_total"] = tabela[colunas_gasto].sum(axis=1)

    tabela["churn"] = (tabela["frequencia_total"] < FREQUENCIA_MINIMA_ATIVO).astype(int)

    return tabela
