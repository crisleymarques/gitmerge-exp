from .utils import add_row_to_dataframe, handle_error, logger
from .prompt import create_prompt
from typing import Tuple

import pandas as pd
import time


def process_dataframe(df, generate_content_fn, start_index=0, max_requests=None, wait_time=20) -> Tuple[pd.DataFrame, int]:
    """
    Generates conflict resolutions for a DataFrame of conflict information.

    Args:
        df: DataFrame with columns 'commit_sha', 'conflict_tuple', 'commit_message'
        generate_content_fn: Function to generate resolution content from a prompt
        start_index: Starting index for processing (default: 0)
        max_requests: Max number of rows to process (default: all remaining)
        wait_time: Seconds to wait between requests (default: 20)

    Returns:
        (DataFrame with resolutions, ending index)
    """
    res_df = pd.DataFrame(columns=['id', 'commit_sha', 'conflict_resolution'])
    total_rows = len(df)
    
    if max_requests is None:
        max_requests = total_rows - start_index
    
    end_index = min(start_index + max_requests, total_rows)
    
    for index in range(start_index, end_index):
        row = df.iloc[index]
        
        logger.section(f"LINHA {index + 1}/{end_index}")
        logger.info(f"Commit SHA: {row['commit_sha']}")
        
        if 'commit_message' in row and row['commit_message']:
            commit_msg = row['commit_message']
            if len(commit_msg) > 80:
                commit_msg = commit_msg[:77] + "..."
            logger.info(f"Mensagem: {commit_msg}")
        
        try:
            prompt = create_prompt(row['conflict_tuple'], row['commit_message'])
            
            logger.info(f"Processando...")
            start_time = time.time()
            response_text = generate_content_fn(prompt)
            elapsed_time = time.time() - start_time
            
            logger.success(f"Resolvido em {elapsed_time:.2f}s")
            
            res_df = add_row_to_dataframe(res_df, row['commit_sha'], response_text, row['id'])
            
        except Exception as e:
            logger.error(f"Erro no processamento")
            res_df = handle_error(res_df, row, e)
        
        if index < end_index - 1:
            logger.info(f"Aguardando próxima linha...")
            time.sleep(wait_time)
    
    return res_df, end_index

