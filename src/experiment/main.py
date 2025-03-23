from .conflict_resolution_generator import process_dataframe
from .config.cli_config import setup_cli_parser, get_llm_config_from_args
from .llm.llm_client import LLMClient
from .utils import get_project_root, logger
from datetime import datetime
from glob import glob

import pandas as pd
import argparse
import logging
import json
import time
import os


INPUT_DATA_DIR = "data/dataset"
INPUT_DATA_FILE = "elastic/elastic_train_conflicts.jsonl"
OUTPUT_DIR = "data/output/2-elastic_qwen_20250316"

EXPERIMENT_NROWS = 2245
EXPERIMENT_START_INDEX = 1300
EXPERIMENT_MAX_REQUESTS = 1500
EXPERIMENT_CHECKPOINT_INTERVAL = 50
EXPERIMENT_WAIT_TIME = 30

REPOSITORY_NAME = "elastic"

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
    
    input_path = os.path.join(project_root, INPUT_DATA_DIR, INPUT_DATA_FILE)
    
    start_time = time.time()
    df = pd.read_json(open(input_path, 'r'), lines=True, nrows=nrows)
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
    
    output_dir = os.path.join(project_root, OUTPUT_DIR)
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
            "repository_name": REPOSITORY_NAME,
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
                end_index - current_index,
                wait_time=EXPERIMENT_WAIT_TIME
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
                
                # Aguardar entre lotes para evitar atingir rate limits
                if current_index < min(total_rows, start_index + max_requests):
                    logger.info(f"Aguardando {EXPERIMENT_WAIT_TIME}s antes do próximo lote...")
                    time.sleep(EXPERIMENT_WAIT_TIME)
            
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
    
    df = load_input_data(project_root, nrows=EXPERIMENT_NROWS)
    
    result_df, last_processed_index = process_data(
        df, 
        llm_client.generate_content,
        start_index=EXPERIMENT_START_INDEX,
        max_requests=EXPERIMENT_MAX_REQUESTS,
        checkpoint_interval=EXPERIMENT_CHECKPOINT_INTERVAL,
        project_root=project_root,
        llm_config=llm_config
    )
    
    save_results(result_df, llm_config, project_root, last_processed_index)
    
    elapsed_time = time.time() - start_time
    logger.section("Concluído")
    logger.success(f"Tempo total: {elapsed_time:.2f}s")


def regenerate_failed_resolutions(input_file, llm_client, wait_time=EXPERIMENT_WAIT_TIME, output_file=None, save_to_original=False):
    """
    Regenera as resoluções de conflitos que falharam devido a erros na API do LLM.
    
    Args:
        input_file (str): Caminho para o arquivo JSON com os resultados
        llm_client (LLMClient): Cliente LLM configurado
        wait_time (int): Tempo de espera entre requisições
        output_file (str, optional): Caminho para salvar o novo arquivo
        save_to_original (bool): Se True, substitui o arquivo original
        
    Returns:
        str: Caminho do arquivo de saída
    """
    logger.section(f"Regenerando resoluções com erro em: {input_file}")
    
    # Carregar resultados
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'results' in data:
            metadata = data.get('metadata', {})
            results = data['results']
        else:
            metadata = {}
            results = data
    except Exception as e:
        logger.error(f"Erro ao carregar o arquivo: {str(e)}")
        return None
    
    # Identificar resoluções com erro
    failed_indices = []
    for i, result in enumerate(results):
        resolution = result.get('conflict_resolution', '')
        # Verificar se contém mensagens de erro
        if isinstance(resolution, str) and (
            resolution.startswith("Erro ao gerar") or 
            "Error generating content" in resolution or 
            "RateLimitError" in resolution or
            "rate_limit_exceeded" in resolution
        ):
            failed_indices.append(i)
    
    logger.info(f"Encontradas {len(failed_indices)} resoluções com erro")
    
    if not failed_indices:
        logger.info("Nenhuma resolução com erro encontrada. Nada a fazer.")
        return input_file
    
    # Carregar dataset original para obter os dados do conflito
    project_root = get_project_root()
    repository_name = metadata.get('repository_name', 'elastic')
    
    data_path = os.path.join(project_root, INPUT_DATA_DIR, repository_name, f"{repository_name}_train_conflicts.jsonl")
    if not os.path.exists(data_path):
        # Tentar encontrar um arquivo JSONL correspondente
        data_dir = os.path.join(project_root, INPUT_DATA_DIR, repository_name)
        if os.path.exists(data_dir):
            jsonl_files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]
            if jsonl_files:
                data_path = os.path.join(data_dir, jsonl_files[0])
            else:
                logger.error(f"Nenhum arquivo JSONL encontrado em {data_dir}")
                return None
        else:
            logger.error(f"Diretório do dataset não encontrado: {data_dir}")
            return None
    
    # Carregar o dataset original
    original_df = pd.read_json(data_path, lines=True)
    
    # Regenerar as resoluções com erro
    for idx, result_idx in enumerate(failed_indices):
        result = results[result_idx]
        result_id = result.get('id')
        
        logger.info(f"Processando {idx+1}/{len(failed_indices)}: ID {result_id}")
        
        # Encontrar dados originais
        original_data = original_df[original_df['id'] == result_id]
        if len(original_data) == 0:
            logger.warning(f"ID {result_id} não encontrado no dataset original. Pulando.")
            continue
        
        original_row = original_data.iloc[0]
        
        try:
            from .prompt import create_prompt
            prompt = create_prompt(original_row['conflict_tuple'], original_row['commit_message'])
            
            logger.info(f"Gerando nova resolução...")
            start_time = time.time()
            response_text = llm_client.generate_content(prompt)
            elapsed_time = time.time() - start_time
            
            logger.success(f"Resolução gerada em {elapsed_time:.2f}s")
            
            # Atualizar o resultado
            results[result_idx]['conflict_resolution'] = response_text
            
        except Exception as e:
            logger.error(f"Erro ao gerar resolução para ID {result_id}: {str(e)}")
        
        # Aguardar entre requisições
        if idx < len(failed_indices) - 1:
            logger.info(f"Aguardando {wait_time}s antes da próxima requisição...")
            time.sleep(wait_time)
    
    # Preparar caminho de saída
    if not output_file:
        if save_to_original:
            output_file = input_file
        else:
            base_name, ext = os.path.splitext(input_file)
            output_file = f"{base_name}_fixed{ext}"
    
    # Atualizar metadata
    metadata.update({
        'regeneration_timestamp': datetime.now().isoformat(),
        'total_records': len(results)
    })
    
    # Salvar resultados
    output_data = {
        'metadata': metadata,
        'results': results
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.success(f"Resultados salvos em: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Erro ao salvar resultados: {str(e)}")
        return None

def regenerate_main():
    """
    Função principal para regeneração de resoluções com erro.
    Esta função pode ser chamada diretamente para regenerar resoluções.
    """
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='Regenera resoluções de conflitos com erro.')
    
    # Argumentos para o arquivo de entrada
    parser.add_argument('--input', type=str, required=True,
                        help='Caminho para o arquivo JSON com os resultados ou padrão glob')
    parser.add_argument('--output', type=str, 
                        help='Caminho para salvar o novo arquivo JSON (opcional)')
    parser.add_argument('--wait-time', type=int, default=EXPERIMENT_WAIT_TIME,
                        help=f'Tempo de espera entre requisições (padrão: {EXPERIMENT_WAIT_TIME}s)')
    parser.add_argument('--glob', action='store_true',
                        help='Tratar o input como um padrão glob e processar múltiplos arquivos')
    parser.add_argument('--overwrite', action='store_true',
                        help='Sobrescrever o arquivo original em vez de criar um novo')
    
    # Adicionar argumentos do LLM
    llm_parser = setup_cli_parser()
    for action in llm_parser._actions:
        if action.dest != 'help':
            parser.add_argument(
                *[opt for opt in action.option_strings], 
                dest=action.dest,
                default=action.default,
                help=action.help,
                choices=action.choices
            )
    
    args = parser.parse_args()
    
    # Configurar cliente LLM
    llm_config = get_llm_config_from_args(args)
    llm_client = LLMClient(llm_config)
    logger.info(f"LLM configurado: {llm_config.provider}, modelo: {llm_config.model}")
    
    # Processar arquivos
    if args.glob:
        input_files = glob(args.input)
        logger.info(f"Encontrados {len(input_files)} arquivos com o padrão '{args.input}'")
        
        for input_file in input_files:
            output_file = args.output
            if not output_file and not args.overwrite:
                base_name, ext = os.path.splitext(input_file)
                output_file = f"{base_name}_fixed{ext}"
            
            regenerate_failed_resolutions(
                input_file=input_file,
                llm_client=llm_client,
                wait_time=args.wait_time,
                output_file=output_file,
                save_to_original=args.overwrite
            )
    else:
        regenerate_failed_resolutions(
            input_file=args.input,
            llm_client=llm_client,
            wait_time=args.wait_time,
            output_file=args.output,
            save_to_original=args.overwrite
        )
    
    logger.section("Processamento concluído")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.error("Processamento interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro: {str(e)}")