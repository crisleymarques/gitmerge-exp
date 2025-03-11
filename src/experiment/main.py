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


def save_checkpoint(result_df, llm_config, project_root, last_processed_index, is_final=False):
    """
    Salva um checkpoint dos resultados em um arquivo JSON com metadados.
    
    Args:
        result_df (pandas.DataFrame): DataFrame com os resultados
        llm_config (LLMConfig): Configuração do LLM
        project_root (str): Caminho raiz do projeto
        last_processed_index (int): Índice do último item processado
        is_final (bool): Indica se é o salvamento final ou um checkpoint intermediário
    """
    logger.section("Salvando " + ("resultados finais" if is_final else "checkpoint"))
    
    output_dir = os.path.join(project_root, "data", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = llm_config.model.split('/')[-1].lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if is_final:
        filename = f"solved_conflicts_{model_name}_{timestamp}.json"
    else:
        filename = f"checkpoint_{model_name}_{timestamp}_idx{last_processed_index}.json"
    
    output_path = os.path.join(output_dir, filename)
    
    output_data = {
        "metadata": {
            "provider": llm_config.provider,
            "model": llm_config.model,
            "timestamp": datetime.now().isoformat(),
            "total_records": len(result_df),
            "last_processed_index": last_processed_index,
            "is_checkpoint": not is_final
        },
        "results": json.loads(result_df.to_json(orient="records"))
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.success(f"Resultados salvos em: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Erro ao salvar resultados: {str(e)}")
        return None


def save_results(result_df, llm_config, project_root, last_processed_index=None):
    """
    Salva os resultados finais em um arquivo JSON com metadados.
    
    Args:
        result_df (pandas.DataFrame): DataFrame com os resultados
        llm_config (LLMConfig): Configuração do LLM
        project_root (str): Caminho raiz do projeto
        last_processed_index (int, optional): Índice do último item processado
    """
    return save_checkpoint(result_df, llm_config, project_root, last_processed_index, is_final=True)


def process_data(df, generate_content_fn, start_index=0, max_requests=1500, checkpoint_interval=50, project_root=None, llm_config=None):
    """
    Processa os dados usando a função de geração de conteúdo.
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados de entrada
        generate_content_fn (callable): Função para gerar conteúdo
        start_index (int): Índice inicial para processamento
        max_requests (int): Número máximo de requisições
        checkpoint_interval (int): Intervalo de registros para salvar checkpoint
        project_root (str): Caminho raiz do projeto (necessário para salvar checkpoints)
        llm_config (LLMConfig): Configuração do LLM (necessário para salvar checkpoints)
        
    Returns:
        tuple: (result_df, last_processed_index) - DataFrame com resultados e último índice processado
    """
    logger.section("Processando dados")
    
    total_rows = len(df)
    logger.info(f"Iniciando processamento de {min(max_requests, total_rows - start_index)} registros")
    
    start_time = time.time()
    
    batch_size = min(checkpoint_interval, max_requests)
    
    result_df = pd.DataFrame()
    current_index = start_index
    
    try:
        while current_index < min(total_rows, start_index + max_requests):
            end_index = min(current_index + batch_size, start_index + max_requests)
            
            logger.info(f"Processando lote de {current_index} até {end_index-1}")
            
            batch_df, last_idx = process_dataframe(
                df, 
                generate_content_fn, 
                current_index, 
                end_index - current_index
            )
            
            current_index = end_index
            
            if not batch_df.empty:
                if result_df.empty:
                    result_df = batch_df
                else:
                    result_df = pd.concat([result_df, batch_df], ignore_index=True)
            
            if project_root and llm_config and not result_df.empty:
                checkpoint_path = save_checkpoint(
                    result_df, 
                    llm_config, 
                    project_root, 
                    end_index,
                    is_final=False
                )
                logger.info(f"Checkpoint salvo no índice {end_index}: {checkpoint_path}")
            
            if last_idx < end_index - 1:
                logger.warning(f"Processamento interrompido no índice {last_idx}")
                break
                
    except KeyboardInterrupt:
        logger.warning("Processamento interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro durante processamento: {str(e)}")
    
    elapsed_time = time.time() - start_time
    last_processed_index = current_index - 1 if current_index > start_index else start_index
    logger.success(f"Processamento concluído em {elapsed_time:.2f}s. Processados {len(result_df)} registros até o índice {last_processed_index}")
    
    return result_df, last_processed_index


def main():
    logging.basicConfig(level=logging.WARNING)
    
    start_time = time.time()
    logger.section("Iniciando processamento")
    
    llm_client, llm_config = setup_llm()
    
    project_root = get_project_root()
    
    df = load_input_data(project_root, nrows=1300)
    
    # Processar os dados com checkpoints a cada 50 itens
    result_df, last_processed_index = process_data(
        df, 
        llm_client.generate_content,
        start_index=0,
        checkpoint_interval=50,
        project_root=project_root,
        llm_config=llm_config
    )
    
    save_results(result_df, llm_config, project_root, last_processed_index)
    
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