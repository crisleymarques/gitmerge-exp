import pandas as pd
import colorlog
import logging
import os


def setup_logger():
    """
    Configura o logger com formatação colorida e níveis personalizados.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    
    logger = logging.getLogger('conflict_resolver')
    logger.setLevel(logging.INFO)
    
    logger.propagate = False
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(message)s%(reset)s",
        log_colors={
            'INFO': 'blue',
            'SUCCESS': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)
    
    logging.SUCCESS = 25  # Entre INFO (20) e WARNING (30)
    logging.addLevelName(logging.SUCCESS, 'SUCCESS')
    
    def success(self, message, *args, **kwargs):
        self.log(logging.SUCCESS, message, *args, **kwargs)
    
    logging.Logger.success = success
    
    def section(self, title):
        if title.startswith("LINHA "):
            self.info("\n" + "=" * 50)
            self.info(f">>> {title} <<<")
            self.info("=" * 50)
        else:
            separator = "=" * 50
            self.info(f"\n{separator}")
            self.info(f"{title}")
            self.info(f"{separator}")
    
    logging.Logger.section = section
    
    logging.getLogger('litellm').setLevel(logging.WARNING)
    
    return logger


logger = setup_logger()


def add_row_to_dataframe(df, commit_sha, conflict_resolution, index, total_rows):
    """
    Adiciona uma nova linha ao DataFrame de resultados.
    
    Args:
        df: DataFrame de resultados
        commit_sha: SHA do commit
        conflict_resolution: Resolução do conflito
        index: Índice atual
        total_rows: Total de linhas
        
    Returns:
        DataFrame atualizado
    """
    new_row = pd.DataFrame([{
        'commit_sha': commit_sha,
        'conflict_resolution': conflict_resolution
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    return df


def handle_error(df, row, error):
    """
    Trata erros durante o processamento e adiciona uma linha de erro ao DataFrame.
    
    Args:
        df: DataFrame de resultados
        row: Linha que causou o erro
        error: Exceção capturada
        index: Índice atual
        
    Returns:
        DataFrame atualizado
    """
    new_row = pd.DataFrame([{
        'commit_sha': row['commit_sha'],
        'conflict_resolution': f"Erro ao gerar: {error}"
    }])

    return pd.concat([df, new_row], ignore_index=True)


def get_project_root():
    """Get the absolute path to the project root directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))