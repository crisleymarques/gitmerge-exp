#!/usr/bin/env python3

"""
Script simplificado para regenerar resoluções de conflitos que falharam devido a erros da API.

Uso:
    python -m src.regenerate_conflicts --input path/to/results.json

Argumentos:
    --input: Caminho para o arquivo JSON com os resultados (pode ser um padrão glob com --glob)
    --output: (Opcional) Caminho para salvar o novo arquivo JSON
    --glob: Tratar o input como um padrão glob para processar múltiplos arquivos
    --overwrite: Sobrescrever os arquivos originais em vez de criar novos
"""

import json
import os
import sys
import time
import argparse
import pandas as pd
from datetime import datetime
import logging
from glob import glob

# Importar recursos necessários do projeto
from .llm.llm_client import LLMClient
from .config.cli_config import setup_cli_parser, get_llm_config_from_args
from .prompt import create_prompt
from .utils import logger, get_project_root

# Configurações padrão
DEFAULT_WAIT_TIME = 60  # Tempo de espera entre requisições para evitar rate limits

def setup_argument_parser():
    """Configura o parser de argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description='Regenera resoluções de conflitos com erro.')
    
    # Argumentos principais
    parser.add_argument('--input', type=str, required=True,
                        help='Caminho para o arquivo JSON com os resultados')
    parser.add_argument('--output', type=str, 
                        help='Caminho para salvar o novo arquivo JSON (opcional)')
    parser.add_argument('--wait-time', type=int, default=DEFAULT_WAIT_TIME,
                        help=f'Tempo de espera entre requisições (padrão: {DEFAULT_WAIT_TIME}s)')
    parser.add_argument('--glob', action='store_true',
                        help='Tratar o input como um padrão glob e processar múltiplos arquivos')
    parser.add_argument('--overwrite', action='store_true',
                        help='Sobrescrever o arquivo original em vez de criar um novo')
    
    # Adicionar argumentos do LLM do CLI original
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
    
    return parser

def load_results(input_file):
    """Carrega os resultados do arquivo JSON."""
    logger.section(f"Carregando resultados de {input_file}")
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'results' in data:
            # Formato com metadata
            metadata = data.get('metadata', {})
            results = data['results']
        else:
            # Lista direta de resultados
            metadata = {}
            results = data
            
        logger.success(f"Carregados {len(results)} registros")
        return metadata, results
    
    except Exception as e:
        logger.error(f"Erro ao carregar o arquivo: {str(e)}")
        sys.exit(1)

def find_dataset_file(repository_name):
    """Encontra o arquivo de dataset para o repositório especificado."""
    project_root = get_project_root()
    
    # Caminho padrão esperado
    data_path = os.path.join(project_root, 'data', 'dataset', repository_name, f"{repository_name}_train_conflicts.jsonl")
    
    if os.path.exists(data_path):
        return data_path
    
    # Buscar por alternativas se o padrão não existir
    data_dir = os.path.join(project_root, 'data', 'dataset', repository_name)
    if os.path.exists(data_dir):
        jsonl_files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]
        if jsonl_files:
            return os.path.join(data_dir, jsonl_files[0])
    
    logger.error(f"Não foi possível encontrar o dataset para o repositório: {repository_name}")
    return None

def identify_failed_resolutions(results):
    """Identifica as resoluções que falharam devido a erros."""
    failed_indices = []
    for i, result in enumerate(results):
        resolution = result.get('conflict_resolution', '')
        
        # Verificar se a string contém indicações de erro
        if isinstance(resolution, str) and (
            resolution.startswith("Erro ao gerar")
        ):
            failed_indices.append(i)
    
    return failed_indices

def process_file(input_file, llm_client, args):
    """Processa um único arquivo de resultados."""
    # Carregar resultados
    metadata, results = load_results(input_file)
    
    # Identificar resoluções com erro
    failed_indices = identify_failed_resolutions(results)
    
    if not failed_indices:
        logger.info(f"Nenhuma resolução com erro encontrada em {input_file}. Nada a fazer.")
        return
    
    logger.info(f"Encontradas {len(failed_indices)} resoluções com erro")
    
    # Determinar o repositório e carregar o dataset original
    repository_name = metadata.get('repository_name', 'elastic')
    dataset_file = find_dataset_file(repository_name)
    
    if not dataset_file:
        logger.error(f"Não foi possível processar o arquivo {input_file} sem o dataset original")
        return
    
    # Carregar o dataset original
    dataset_df = pd.read_json(dataset_file, lines=True)
    logger.success(f"Dataset original carregado: {dataset_file}")
    
    # Regenerar resoluções com erro
    updated_count = 0
    total = len(failed_indices)
    
    for idx, result_idx in enumerate(failed_indices):
        result = results[result_idx]
        result_id = result.get('id')
        
        logger.info(f"Processando {idx+1}/{total}: ID {result_id}")
        
        # Encontrar os dados originais do conflito
        original_data = dataset_df[dataset_df['id'] == result_id]
        
        if len(original_data) == 0:
            logger.warning(f"ID {result_id} não encontrado no dataset original. Pulando.")
            continue
        
        original_row = original_data.iloc[0]
        
        try:
            # Criar prompt e gerar nova resolução
            prompt = create_prompt(original_row['conflict_tuple'], original_row['commit_message'])
            
            logger.info(f"Gerando nova resolução...")
            start_time = time.time()
            response_text = llm_client.generate_content(prompt)
            elapsed_time = time.time() - start_time
            
            logger.success(f"Resolução gerada em {elapsed_time:.2f}s")
            
            # Atualizar o resultado
            results[result_idx]['conflict_resolution'] = response_text
            updated_count += 1
            
        except Exception as e:
            logger.error(f"Erro ao gerar resolução: {str(e)}")
        
        # Aguardar entre requisições para evitar rate limits
        if idx < total - 1:
            logger.info(f"Aguardando {args.wait_time}s antes da próxima requisição...")
            time.sleep(args.wait_time)
    
    logger.success(f"Regeneradas {updated_count} resoluções com sucesso")
    
    # Determinar caminho de saída
    output_file = args.output
    if not output_file:
        if args.overwrite:
            output_file = input_file
        else:
            base_name, ext = os.path.splitext(input_file)
            output_file = f"{base_name}_fixed{ext}"
    
    # Atualizar metadata
    metadata.update({
        'regeneration_timestamp': datetime.now().isoformat(),
        'regenerated_count': updated_count,
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

def main():
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Configurar parser de argumentos
    parser = setup_argument_parser()
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
            logger.section(f"Processando arquivo: {input_file}")
            process_file(input_file, llm_client, args)
    else:
        process_file(args.input, llm_client, args)
    
    logger.section("Processamento concluído")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.error("Processamento interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erro: {str(e)}")
        sys.exit(1) 