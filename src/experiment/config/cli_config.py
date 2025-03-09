from ..llm.llm_config import (
    LLMConfig,
    create_config,
    get_default_config
)

import argparse


def setup_cli_parser() -> argparse.ArgumentParser:
    """
    Set up the command line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description='Process conflicts using LLM')
    parser.add_argument('--provider', type=str, default=None,
                      help='LLM provider (google, groq, maritaca)')
    parser.add_argument('--model', type=str, default=None,
                      help='Model name (e.g., gemini-2.0-flash, qwen-2.5-coder-32b, sabia-3)')
    return parser


def get_llm_config_from_args(args: argparse.Namespace) -> LLMConfig:
    """
    Get LLM configuration based on command line arguments.
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Returns:
        LLMConfig: Configuration for the LLM client
    """
    if args.provider and args.model:
        return create_config(args.provider, args.model)
    else:
        return get_default_config()