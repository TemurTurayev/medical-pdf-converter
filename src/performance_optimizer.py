import os
import time
import cProfile
import threading
from typing import Dict, List, Any, Callable
from functools import lru_cache
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import logging
from dataclasses import dataclass
from queue import Queue
import numpy as np

@dataclass
class ProcessingTask:
    """Class representing a processing task"""
    task_id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: int = 0

@dataclass
class PerformanceMetrics:
    """Class for storing performance metrics"""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    task_type: str

class PerformanceOptimizer:
    def __init__(self, max_workers: int = None, cache_size: int = 128):
        """Initialize performance optimizer"""
        self.max_workers = max_workers or mp.cpu_count()
        self.cache_size = cache_size
        self.metrics_history: List[PerformanceMetrics] = []
        self.task_queue = Queue()
        self.setup_logging()
        
        # Настройка кэша
        self.cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Мониторинг ресурсов
        self.process = psutil.Process()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('performance.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    @lru_cache(maxsize=128)
    def cached_process(self, data_hash: str, *args, **kwargs):
        """Cached version of processing function"""
        return self.process_data(*args, **kwargs)

    def process_batch(self, tasks: List[ProcessingTask], use_processes: bool = False) -> Dict:
        """Process multiple tasks in parallel"""
        start_time = time.time()
        results = {}
        
        if use_processes:
            executor_class = ProcessPoolExecutor
        else:
            executor_class = ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            futures = {}
            for task in tasks:
                future = executor.submit(
                    self._execute_task,
                    task.function,
                    *task.args,
                    **task.kwargs
                )
                futures[future] = task.task_id
            
            for future in futures:
                task_id = futures[future]
                try:
                    results[task_id] = future.result()
                except Exception as e:
                    self.logger.error(f"Error processing task {task_id}: {str(e)}")
                    results[task_id] = None
        
        execution_time = time.time() - start_time
        self._record_metrics(execution_time, 'batch_processing')
        
        return results

    def _execute_task(self, func: Callable, *args, **kwargs) -> Any:
        """Execute single task with profiling"""
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
        
        return result

    def optimize_memory(self, data: Any) -> Any:
        """Optimize memory usage for data structures"""
        if isinstance(data, dict):
            return self._optimize_dict(data)
        elif isinstance(data, list):
            return self._optimize_list(data)
        elif isinstance(data, str):
            return self._optimize_string(data)
        elif isinstance(data, np.ndarray):
            return self._optimize_array(data)
        return data

    def _optimize_dict(self, data: Dict) -> Dict:
        """Optimize dictionary memory usage"""
        optimized = {}
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                value = self.optimize_memory(value)
            optimized[key] = value
        return optimized

    def _optimize_list(self, data: List) -> List:
        """Optimize list memory usage"""
        return [self.optimize_memory(item) if isinstance(item, (dict, list)) else item
                for item in data]

    def _optimize_string(self, data: str) -> str:
        """Optimize string memory usage using interning"""
        return sys.intern(data) if len(data) < 1000 else data

    def _optimize_array(self, data: np.ndarray) -> np.ndarray:
        """Optimize numpy array memory usage"""
        # Оптимизация типа данных
        if data.dtype == np.float64:
            return data.astype(np.float32, copy=False)
        elif data.dtype == np.int64:
            return data.astype(np.int32, copy=False)
        return data

    def _record_metrics(self, execution_time: float, task_type: str):
        """Record performance metrics"""
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_usage=self.process.memory_info().rss / 1024 / 1024,  # MB
            cpu_usage=self.process.cpu_percent(),
            task_type=task_type
        )
        self.metrics_history.append(metrics)
        
        # Удаление старых метрик
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        if not self.metrics_history:
            return {}

        report = {
            'summary': {
                'total_tasks': len(self.metrics_history),
                'avg_execution_time': np.mean([m.execution_time for m in self.metrics_history]),
                'avg_memory_usage': np.mean([m.memory_usage for m in self.metrics_history]),
                'avg_cpu_usage': np.mean([m.cpu_usage for m in self.metrics_history])
            },
            'cache_stats': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_ratio': self.cache_hits / (self.cache_hits + self.cache_misses) if 
                            (self.cache_hits + self.cache_misses) > 0 else 0
            },
            'task_types': {}
        }

        # Статистика по типам задач
        task_types = set(m.task_type for m in self.metrics_history)
        for task_type in task_types:
            type_metrics = [m for m in self.metrics_history if m.task_type == task_type]
            report['task_types'][task_type] = {
                'count': len(type_metrics),
                'avg_execution_time': np.mean([m.execution_time for m in type_metrics]),
                'avg_memory_usage': np.mean([m.memory_usage for m in type_metrics]),
                'avg_cpu_usage': np.mean([m.cpu_usage for m in type_metrics])
            }

        return report

    def optimize_image_processing(self, image: np.ndarray, target_size: tuple = None) -> np.ndarray:
        """Optimize image for processing"""
        # Оптимизация размера
        if target_size and (image.shape[0] > target_size[0] or image.shape[1] > target_size[1]):
            scale = min(target_size[0] / image.shape[0], target_size[1] / image.shape[1])
            new_size = tuple(int(dim * scale) for dim in image.shape[:2])
            image = cv2.resize(image, new_size[::-1])

        # Оптимизация типа данных
        if image.dtype == np.float64:
            image = image.astype(np.float32)

        return image

    def clear_cache(self):
        """Clear processing cache"""
        self.cache.clear()
        self.cached_process.cache_clear()

    def schedule_task(self, task: ProcessingTask):
        """Schedule task for processing"""
        self.task_queue.put((task.priority, task))

    def process_queue(self, batch_size: int = 10):
        """Process tasks from queue in batches"""
        while not self.task_queue.empty():
            batch = []
            for _ in range(batch_size):
                if self.task_queue.empty():
                    break
                priority, task = self.task_queue.get()
                batch.append(task)

            if batch:
                results = self.process_batch(batch)
                for task, result in zip(batch, results.values()):
                    self.logger.info(f"Completed task {task.task_id}")

    def __del__(self):
        """Cleanup resources"""
        self.clear_cache()
