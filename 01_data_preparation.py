import pandas as pd
import json
import os

# Define os nomes dos arquivos para facilitar a manutenção
JSON_FILE = "TelecomX_Data.json"
CSV_OUTPUT_FILE = "telecom_data_tratado.csv"

def run_data_preparation():
    """
    Carrega os dados de um arquivo JSON, realiza a limpeza e o pré-processamento,
    e salva o resultado em um arquivo CSV.
    """
    # Constrói os caminhos relativos ao diretório do script
    script_dir = os.path.dirname(__file__)
    json_file_path = os.path.join(script_dir, JSON_FILE)
    csv_output_path = os.path.join(script_dir, CSV_OUTPUT_FILE)

    print("Iniciando a Etapa 1: Preparação dos Dados...")

    # 1. Carregar os dados do arquivo JSON
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Arquivo JSON '{JSON_FILE}' carregado com sucesso.")
    except FileNotFoundError:
        print(f"Erro: O arquivo '{json_file_path}' não foi encontrado.")
        return
    except json.JSONDecodeError:
        print(f"Erro: O arquivo '{json_file_path}' não é um JSON válido.")
        return

    # 2. Normalizar a estrutura do JSON para um DataFrame
    df_normalized = pd.json_normalize(data)
    print("Estrutura JSON normalizada para DataFrame.")

    # Renomear colunas para remover prefixos desnecessários
    df_normalized.columns = [
        col.replace('customer.', '').replace('phone.', '').replace('internet.', '').replace('account.', '').replace('Charges.', '')
        for col in df_normalized.columns
    ]
    print("Colunas renomeadas.")

    # 3. Limpeza e Pré-processamento dos Dados
    print("Iniciando limpeza e pré-processamento...")

    # Corrigir a coluna 'Total': converter para numérico e tratar valores ausentes
    df_normalized['Total'] = pd.to_numeric(df_normalized['Total'], errors='coerce')
    df_normalized['Total'].fillna(0, inplace=True)
    print("Coluna 'Total' corrigida.")

    # Tratar valores ausentes ou vazios na coluna 'Churn'
    df_normalized.dropna(subset=['Churn'], inplace=True)
    df_normalized = df_normalized[df_normalized['Churn'] != ''].copy()
    print("Valores ausentes em 'Churn' tratados.")

    # Converter a coluna 'Churn' para formato numérico (0 ou 1)
    df_normalized['Churn'] = df_normalized['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    print("Coluna 'Churn' convertida para formato numérico.")

    # Padronizar valores categóricos para consistência
    df_normalized.replace(
        {'No internet service': 'No', 'No phone service': 'No'},
        inplace=True
    )
    print("Valores categóricos padronizados.")

    # 4. Salvar o DataFrame tratado em um arquivo CSV
    df_normalized.to_csv(csv_output_path, index=False)
    print(f"\nArquivo tratado salvo com sucesso em: '{csv_output_path}'")
    print("Etapa 1 concluída.")

if __name__ == '__main__':
    run_data_preparation()
