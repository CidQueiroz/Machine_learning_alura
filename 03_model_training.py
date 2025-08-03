import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define os nomes dos arquivos
X_PROCESSED_FILE = "X_processed.csv"
Y_TARGET_FILE = "y_target.csv"

def evaluate_model(model, X_test, y_test, model_name):
    """
    Avalia um modelo de classificação e exibe as métricas de performance,
    incluindo acurácia, relatório de classificação e matriz de confusão.
    """
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Imprimir resultados
    print(f"--- Avaliação do Modelo: {model_name} ---")
    print(f"Acurácia: {accuracy:.4f}")
    print("\nRelatório de Classificação:")
    print(report)
    print("\nMatriz de Confusão:")
    print(cm)
    
    # Visualizar a Matriz de Confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.ylabel("Verdadeiro")
    plt.xlabel("Previsto")
    plt.show()

def run_model_training():
    """
    Carrega os dados processados, treina e avalia os modelos de Regressão Logística
    e Random Forest, e analisa a importância das features.
    """
    # Constrói os caminhos relativos
    script_dir = os.path.dirname(__file__)
    X_path = os.path.join(script_dir, X_PROCESSED_FILE)
    y_path = os.path.join(script_dir, Y_TARGET_FILE)

    print("\nIniciando a Etapa 3: Treinamento e Avaliação dos Modelos...")

    # 1. Carregar os dados
    try:
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path)
        print("Dados processados carregados com sucesso.")
    except FileNotFoundError:
        print(f"Erro: Arquivos de dados não encontrados. Execute as etapas anteriores.")
        return

    # 2. Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print("Dados divididos em conjuntos de treino e teste.")

    # 3. Treinar e Avaliar Modelos
    
    # Modelo 1: Regressão Logística
    print("\nTreinando o modelo de Regressão Logística...")
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    logreg.fit(X_train, y_train.values.ravel())
    evaluate_model(logreg, X_test, y_test, "Regressão Logística")

    # Modelo 2: Random Forest
    print("\nTreinando o modelo Random Forest...")
    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
    rf.fit(X_train, y_train.values.ravel())
    evaluate_model(rf, X_test, y_test, "Random Forest")

    # 4. Análise de Importância das Variáveis (Feature Importance) com Random Forest
    print("\n--- Análise de Importância das Variáveis (Random Forest) ---")
    feature_importances = rf.feature_importances_
    features = X.columns

    # Criar um DataFrame para visualização
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

    # Exibir as top 10 features mais importantes
    top_n = 10
    print(f"\nAs {top_n} variáveis mais importantes:")
    print(feature_importance_df.head(top_n))

    # Visualizar a importância das variáveis
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n), palette="viridis")
    plt.title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.xlabel("Importância")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
    
    print("\nEtapa 3 concluída.")

if __name__ == '__main__':
    run_model_training()
