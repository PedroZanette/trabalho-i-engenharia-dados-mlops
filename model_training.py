"""Ciência de Dados: split treino/teste, treinamento e métricas do modelo.

Modelo escolhido: DecisionTreeClassifier(max_depth=3).
Justificativa: a variável churn foi definida por uma regra de frequência
(frequencia_total < 8); uma árvore de decisão rasa é simples, interpretável
e adequada para demonstrar esse limite de decisão.
"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def treinar_modelo(tabela):
    features = [
        c
        for c in tabela.columns
        if c not in ("id_cliente", "churn")
    ]

    X = tabela[features]
    y = tabela["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    metricas = {
        "acuracia": accuracy_score(y_test, y_pred),
        "matriz_confusao": confusion_matrix(y_test, y_pred),
        "n_treino": len(X_train),
        "n_teste": len(X_test),
    }

    return modelo, features, metricas
