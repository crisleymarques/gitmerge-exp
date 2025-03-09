from .conflict_resolution_generator import process_dataframe
from .config.cli_config import setup_cli_parser, get_llm_config_from_args
from .llm.llm_client import LLMClient
from .utils import get_project_root, logger
from datetime import datetime

import pandas as pd
import logging
import json
import time
import os


def setup_llm():
    """
    Configura e inicializa o cliente LLM com base nos argumentos da linha de comando.
    
    Returns:
        tuple: (llm_client, llm_config) - O cliente LLM e sua configuração
    """
    logger.section("Configuração LLM")
    
    parser = setup_cli_parser()
    args = parser.parse_args()
    
    llm_config = get_llm_config_from_args(args)
    llm_client = LLMClient(llm_config)
    logger.info(f"Provider: {llm_config.provider}, Modelo: {llm_config.model}")
    
    return llm_client, llm_config


def load_input_data(project_root, nrows=1):
    """
    Carrega os dados de entrada do arquivo JSONL.
    
    Args:
        project_root (str): Caminho raiz do projeto
        nrows (int): Número de linhas a serem carregadas
        
    Returns:
        pandas.DataFrame: DataFrame com os dados de entrada
    """
    logger.section("Carregando dados")
    
    input_path = os.path.join(project_root, "data", "eclipse_val_conflicts.jsonl")
    
    start_time = time.time()
    df = pd.read_json(input_path, lines=True, nrows=nrows)
    elapsed_time = time.time() - start_time
    
    logger.success(f"Carregados {len(df)} registros em {elapsed_time:.2f}s")
    
    return df


def process_data(df, generate_content_fn, start_index=0, max_requests=1500):
    """
    Processa os dados usando a função de geração de conteúdo.
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados de entrada
        generate_content_fn (callable): Função para gerar conteúdo
        start_index (int): Índice inicial para processamento
        max_requests (int): Número máximo de requisições
        
    Returns:
        tuple: (result_df, last_processed_index) - DataFrame com resultados e último índice processado
    """
    logger.section("Processando dados")
    
    total_rows = len(df)
    logger.info(f"Iniciando processamento de {min(max_requests, total_rows - start_index)} registros")
    
    start_time = time.time()
    result_df, last_processed_index = process_dataframe(df,
                                                       generate_content_fn, 
                                                       start_index, 
                                                       max_requests)
    elapsed_time = time.time() - start_time
    
    logger.success(f"Processamento concluído em {elapsed_time:.2f}s")
    
    return result_df, last_processed_index


def save_results(result_df, llm_config, project_root):
    """
    Salva os resultados em um arquivo JSON com metadados.
    
    Args:
        result_df (pandas.DataFrame): DataFrame com os resultados
        llm_config (LLMConfig): Configuração do LLM
        project_root (str): Caminho raiz do projeto
    """
    logger.section("Salvando resultados")
    
    output_dir = os.path.join(project_root, "data", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = llm_config.model.split('/')[-1].lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"solved_conflicts_{model_name}_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)
    
    output_data = {
        "metadata": {
            "provider": llm_config.provider,
            "model": llm_config.model,
            "timestamp": datetime.now().isoformat(),
            "total_records": len(result_df)
        },
        "results": json.loads(result_df.to_json(orient="records"))
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.success(f"Resultados salvos em: {output_path}")
    except Exception as e:
        logger.error(f"Erro ao salvar resultados: {str(e)}")


def main():
    logging.basicConfig(level=logging.WARNING)
    
    start_time = time.time()
    logger.section("Iniciando processamento")
    
    # Configurar e inicializar o LLM
    llm_client, llm_config = setup_llm()
    
    # Obter o caminho raiz do projeto
    project_root = get_project_root()
    
    # Carregar os dados de entrada
    df = load_input_data(project_root, nrows=2)
    
    # Processar os dados
    result_df, _ = process_data(df, llm_client.generate_content)
    
    # Salvar os resultados
    save_results(result_df, llm_config, project_root)
    
    elapsed_time = time.time() - start_time
    logger.section("Concluído")
    logger.success(f"Tempo total: {elapsed_time:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.error("Processamento interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro: {str(e)}")