import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Define os nomes dos arquivos
INPUT_CSV_FILE = "telecom_data_tratado.csv"
OUTPUT_X_FILE = "X_processed.csv"
OUTPUT_Y_FILE = "y_target.csv"

def run_feature_engineering():
    """
    Carrega os dados tratados, realiza a engenharia de features (One-Hot Encoding e escalonamento)
    e salva as features processadas (X) e a variável alvo (y) em arquivos separados.
    """
    # Constrói os caminhos relativos
    script_dir = os.path.dirname(__file__)
    input_csv_path = os.path.join(script_dir, INPUT_CSV_FILE)
    output_X_path = os.path.join(script_dir, OUTPUT_X_FILE)
    output_y_path = os.path.join(script_dir, OUTPUT_Y_FILE)

    print("\nIniciando a Etapa 2: Engenharia de Features...")

    # 1. Carregar os dados tratados
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Arquivo '{INPUT_CSV_FILE}' carregado com sucesso.")
    except FileNotFoundError:
        print(f"Erro: O arquivo '{input_csv_path}' não foi encontrado. Execute a preparação de dados primeiro.")
        return

    # Remover a coluna de ID, que não é útil para o modelo
    if 'customerID' in df.columns:
        df_processed = df.drop('customerID', axis=1)
    else:
        df_processed = df

    # 2. Separar as features (X) da variável alvo (y)
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    print("Features (X) e variável alvo (y) separadas.")

    # 3. Encoding de Variáveis Categóricas (One-Hot Encoding)
    X_encoded = pd.get_dummies(X, drop_first=True)
    print("Variáveis categóricas transformadas com One-Hot Encoding.")

    # 4. Escalonamento de Variáveis Numéricas
    # Identificar colunas numéricas para escalonamento
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Inicializar e aplicar o scaler
    scaler = StandardScaler()
    X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])
    print("Variáveis numéricas escalonadas com StandardScaler.")

    # 5. Salvar os dados processados
    X_encoded.to_csv(output_X_path, index=False)
    y.to_csv(output_y_path, index=False, header=['Churn'])

    print(f"\nEngenharia de features concluída!")
    print(f"Features processadas salvas em: '{output_X_path}'")
    print(f"Variável alvo salva em: '{output_y_path}'")
    print("Etapa 2 concluída.")

if __name__ == '__main__':
    run_feature_engineering()
