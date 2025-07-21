"""
real_time_analyzer.py
Real-time analysis and visualization for brain mapping toolkit.
"""
import numpy as np
import matplotlib.pyplot as plt


class RealTimeAnalyzer:
    """Enable real-time brain activity monitoring and analysis."""
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.data_buffer = []

    def stream_data(self, data_source):
        for chunk in data_source:
            self.data_buffer.append(chunk)
            if len(self.data_buffer) > self.buffer_size:
                self.data_buffer.pop(0)

    def real_time_processing(self, data_chunk: np.ndarray):
        return np.mean(data_chunk, axis=0)

    def live_visualization(self, results: dict):
        plt.plot(results['signal'])
        plt.show()
