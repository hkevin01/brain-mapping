"""
Scalability and performance testing utilities for cloud deployment.
"""
import time
import logging


def test_scalability(workflow_func, n_runs=10):
    times = []
    for i in range(n_runs):
        start = time.time()
        workflow_func()
        elapsed = time.time() - start
        times.append(elapsed)
        logging.info("Run %d: %.2fs", i + 1, elapsed)
    avg_time = sum(times) / len(times)
    logging.info("Average run time over %d runs: %.2fs", n_runs, avg_time)
    return avg_time
