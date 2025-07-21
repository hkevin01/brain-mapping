"""
Logging Utilities
================

Logging configuration and utilities for the brain mapping toolkit.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(level: int = logging.INFO, 
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the brain mapping toolkit.
    
    Parameters
    ----------
    level : int, default=logging.INFO
        Logging level
    log_file : str, optional
        Path to log file. If None, logs to console only
    format_string : str, optional
        Custom format string for log messages
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger('brain_mapping')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'brain_mapping') -> logging.Logger:
    """
    Get a logger instance for the specified name.
    
    Parameters
    ----------
    name : str, default='brain_mapping'
        Logger name
        
    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)


# Default logger setup
default_logger = setup_logging() 