"""MLOps & Integração: orquestra pipeline completo e roda inferência."""

import pandas as pd

from data_processing import carregar_dados, processar_dados
from model_training import treinar_modelo


def main():
    df = carregar_dados()
    tabela = processar_dados(df)

    print("=== Engenharia de Dados ===")
    print(f"Transações lidas: {len(df)}")
    print(f"Clientes únicos: {df['id_cliente'].nunique()}")
    print(f"Shape da tabela dinâmica: {tabela.shape}")
    print(f"Distribuição do churn:\n{tabela['churn'].value_counts()}")

    modelo, features, metricas = treinar_modelo(tabela)

    print("\n=== Ciência de Dados ===")
    print(f"Treino: {metricas['n_treino']} | Teste: {metricas['n_teste']}")
    print(f"Acurácia: {metricas['acuracia']:.4f}")
    print(f"Matriz de Confusão:\n{metricas['matriz_confusao']}")

    print("\n=== Inferência: novo cliente ===")
    novo_cliente = {c: 0 for c in features}
    novo_cliente.update(
        {
            "gasto_alimentos": 50.0,
            "freq_alimentos": 1,
            "gasto_roupas": 30.0,
            "freq_roupas": 1,
            "frequencia_total": 2,
            "gasto_total": 80.0,
        }
    )
    novo_cliente_df = pd.DataFrame([novo_cliente])[features]

    previsao = modelo.predict(novo_cliente_df)[0]
    resultado = "CHURN" if previsao == 1 else "NÃO CHURN"

    print(f"Novo cliente: {novo_cliente}")
    print(f"Previsão: {resultado}")


if __name__ == "__main__":
    main()
