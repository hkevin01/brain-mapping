"""
Performance monitoring utilities for cloud workflows.
"""
import time
import logging


def monitor_performance(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    logging.info("Function %s executed in %.2fs", func.__name__, elapsed)
    return result, elapsed
