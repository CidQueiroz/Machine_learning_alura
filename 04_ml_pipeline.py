import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define o nome do arquivo de dados
INPUT_CSV_FILE = "telecom_data_tratado.csv"

def create_pipeline(model_type='logistic_regression'):
    """
    Cria um pipeline de pré-processamento e modelagem.

    Args:
        model_type (str): O tipo de modelo a ser usado no pipeline 
                          ('logistic_regression' ou 'random_forest').

    Returns:
        sklearn.pipeline.Pipeline: O pipeline de ML configurado.
    """
    # Define as colunas categóricas e numéricas com base em tipos de dados comuns
    # Em um cenário real, isso pode ser mais robusto
    categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    numerical_features = ['SeniorCitizen', 'tenure', 'Monthly', 'Total']

    # Cria o transformador de pré-processamento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ], 
        remainder='passthrough'  # Mantém outras colunas, se houver
    )

    # Define o classificador com base na escolha
    if model_type == 'random_forest':
        classifier = RandomForestClassifier(random_state=42, class_weight='balanced')
    else:  # Padrão é Regressão Logística
        classifier = LogisticRegression(random_state=42)

    # Cria o pipeline completo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return pipeline

def run_pipeline():
    """
    Executa o pipeline de ML: carrega os dados, treina o modelo, avalia a performance
    e salva o pipeline treinado em um arquivo.
    """
    # Constrói o caminho relativo para o arquivo de dados
    script_dir = os.path.dirname(__file__)
    input_csv_path = os.path.join(script_dir, INPUT_CSV_FILE)

    print("\nIniciando a Etapa 4: Execução do Pipeline de Machine Learning...")

    # 1. Carregar os dados
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Arquivo '{INPUT_CSV_FILE}' carregado com sucesso.")
    except FileNotFoundError:
        print(f"Erro: O arquivo '{input_csv_path}' não foi encontrado. Execute a preparação de dados.")
        return

    # 2. Separar features (X) e alvo (y)
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn']

    # 3. Dividir em dados de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print("Dados divididos em conjuntos de treino e teste.")

    # 4. Criar e treinar o pipeline com Random Forest (melhor modelo)
    print("\nCriando e treinando o pipeline com Random Forest...")
    rf_pipeline = create_pipeline(model_type='random_forest')
    rf_pipeline.fit(X_train, y_train)
    print("Pipeline treinado com sucesso.")

    # 5. Avaliar o modelo
    y_pred = rf_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n--- Avaliação do Pipeline (Random Forest) ---")
    print(f"Acurácia: {accuracy:.4f}")
    print("\nRelatório de Classificação:")
    print(report)

    # 6. Salvar o pipeline treinado
    pipeline_filename = "random_forest_pipeline.pkl"
    pipeline_path = os.path.join(script_dir, pipeline_filename)
    joblib.dump(rf_pipeline, pipeline_path)
    print(f"\nPipeline salvo com sucesso em: '{pipeline_path}'")
    print("Etapa 4 concluída.")

if __name__ == '__main__':
    run_pipeline()
