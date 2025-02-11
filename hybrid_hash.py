import random
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, List, Tuple, Dict
from dataclasses import dataclass
import json
from datetime import datetime

@dataclass
class BenchmarkStats:
    operation: str
    load_factor: float
    num_operations: int
    total_probes: int
    min_probes: int
    max_probes: int
    avg_probes: float
    std_dev: float
    total_time: float
    operations_per_second: float

    def to_dict(self):
        return {
            'operation': self.operation,
            'load_factor': self.load_factor,
            'num_operations': self.num_operations,
            'total_probes': self.total_probes,
            'min_probes': self.min_probes,
            'max_probes': self.max_probes,
            'avg_probes': self.avg_probes,
            'std_dev': self.std_dev,
            'total_time': self.total_time,
            'operations_per_second': self.operations_per_second
        }

class HashTableBase:
    def __init__(self, size: int):
        self.size = size
        self.inserted = 0

    def load_factor(self) -> float:
        return self.inserted / self.size

    def _calculate_stats(self, operation: str, probes: List[int],
                        time_taken: float) -> BenchmarkStats:
        if not probes:
            return None

        return BenchmarkStats(
            operation=operation,
            load_factor=self.load_factor(),
            num_operations=len(probes),
            total_probes=sum(probes),
            min_probes=min(probes),
            max_probes=max(probes),
            avg_probes=np.mean(probes),
            std_dev=np.std(probes),
            total_time=time_taken,
            operations_per_second=len(probes)/time_taken if time_taken > 0 else 0
        )

class TraditionalHashTable(HashTableBase):
    def __init__(self, size: int):
        super().__init__(size)
        self.table = [None] * size

    def insert(self, key: Any) -> int:
        if self.inserted >= self.size:
            raise Exception("Hash table is full")

        start = hash(key) % self.size
        pos = start
        probes = 0

        while True:
            probes += 1
            if self.table[pos] is None:
                self.table[pos] = key
                self.inserted += 1
                return probes
            pos = (pos + 1) % self.size
            if pos == start:
                raise Exception("Hash table is full")

    def search(self, key: Any) -> int:
        start = hash(key) % self.size
        pos = start
        probes = 0

        while True:
            probes += 1
            if self.table[pos] == key:
                return probes
            if self.table[pos] is None:
                return probes
            pos = (pos + 1) % self.size
            if pos == start:
                return probes

class ElasticHashTable(HashTableBase):
    def __init__(self, size: int, delta: float):
        super().__init__(size)
        self.delta = delta
        self.arrays = self._create_arrays()

    def _create_arrays(self) -> List[List[Any]]:
        arrays = []
        size = self.size
        while size > 1:
            arrays.append([None] * size)
            size = size // 2
        return arrays

    def _probe_sequence(self, key: Any, array_idx: int):
        def hash_func(j):
            return hash(f"{key}:{j}") % len(self.arrays[array_idx])
        return (hash_func(j) for j in range(1, len(self.arrays[array_idx])+1))

    def _f(self, epsilon: float) -> int:
        c = 10
        return int(c * min(math.log(1/epsilon, 2)**2, math.log(1/self.delta, 2)))

    def insert(self, key: Any) -> int:
        if self.inserted >= self.size - math.floor(self.delta * self.size):
            raise Exception("Hash table is full")

        probes = 0
        for i in range(len(self.arrays)):
            epsilon1 = 1 - (self.inserted / len(self.arrays[i]))
            epsilon2 = 1 - (self.inserted / len(self.arrays[i+1])) if i+1 < len(self.arrays) else 1

            if epsilon1 > self.delta/2 and epsilon2 > 0.25:
                for j, pos in enumerate(self._probe_sequence(key, i)):
                    probes += 1
                    if j >= self._f(epsilon1):
                        break
                    if self.arrays[i][pos] is None:
                        self.arrays[i][pos] = key
                        self.inserted += 1
                        return probes
                for pos in self._probe_sequence(key, i+1):
                    probes += 1
                    if self.arrays[i+1][pos] is None:
                        self.arrays[i+1][pos] = key
                        self.inserted += 1
                        return probes
            elif epsilon1 <= self.delta/2:
                for pos in self._probe_sequence(key, i+1):
                    probes += 1
                    if self.arrays[i+1][pos] is None:
                        self.arrays[i+1][pos] = key
                        self.inserted += 1
                        return probes
            elif epsilon2 <= 0.25:
                for pos in self._probe_sequence(key, i):
                    probes += 1
                    if self.arrays[i][pos] is None:
                        self.arrays[i][pos] = key
                        self.inserted += 1
                        return probes

        raise Exception("Failed to insert key")

    def search(self, key: Any) -> int:
        probes = 0
        for i, array in enumerate(self.arrays):
            for j, pos in enumerate(self._probe_sequence(key, i)):
                probes += 1
                if array[pos] == key:
                    return probes
                if array[pos] is None:
                    break
        return probes

class HybridHashTable(HashTableBase):
    def __init__(self, size: int, threshold: float = 0.7):
        super().__init__(size)
        self.threshold = threshold
        self.traditional_table = [None] * size
        self.elastic_table = None
        self.elastic_keys = set()  # Track which keys are in elastic table

    def insert(self, key: Any) -> int:
        probes = 0
        if self.load_factor() < self.threshold:
            # Use traditional insertion
            probes = self._traditional_insert(key)
        else:
            # Initialize elastic table if needed
            if self.elastic_table is None:
                remaining_space = self.size - self.inserted
                self.elastic_table = ElasticHashTable(
                    size=remaining_space * 2,  # Give some extra space
                    delta=0.5
                )
            # Use elastic insertion
            probes = self._elastic_insert(key)
        return probes

    def _traditional_insert(self, key: Any) -> int:
        if self.inserted >= self.size:
            raise Exception("Hash table is full")

        start = hash(key) % self.size
        pos = start
        probes = 0

        while True:
            probes += 1
            if self.traditional_table[pos] is None:
                self.traditional_table[pos] = key
                self.inserted += 1
                return probes
            pos = (pos + 1) % self.size
            if pos == start:
                raise Exception("Hash table is full")

    def _elastic_insert(self, key: Any) -> int:
        probes = self.elastic_table.insert(key)
        self.elastic_keys.add(key)
        self.inserted += 1
        return probes

    def search(self, key: Any) -> int:
        # First check if key is known to be in elastic table
        if key in self.elastic_keys:
            return self.elastic_table.search(key)

        # Search traditional table
        probes = 0
        start = hash(key) % self.size
        pos = start

        while True:
            probes += 1
            if self.traditional_table[pos] == key:
                return probes
            if self.traditional_table[pos] is None:
                # If not found and elastic table exists, search there
                if self.elastic_table is not None:
                    elastic_probes = self.elastic_table.search(key)
                    return probes + elastic_probes
                return probes
            pos = (pos + 1) % self.size
            if pos == start:
                if self.elastic_table is not None:
                    elastic_probes = self.elastic_table.search(key)
                    return probes + elastic_probes
                return probes

class HashTableBenchmark:
    def __init__(self, sizes: List[int], load_factors: List[float]):
        self.sizes = sizes
        self.load_factors = load_factors
        self.results = []

    def run_comprehensive_benchmark(self, num_searches: int = 1000):
        for size in self.sizes:
            for load_factor in self.load_factors:
                print(f"\nBenchmarking size={size}, load_factor={load_factor}")

                # Test all three implementations
                traditional_stats = self._benchmark_implementation(
                    "Traditional", size, load_factor, num_searches)
                elastic_stats = self._benchmark_implementation(
                    "Elastic", size, load_factor, num_searches)
                hybrid_stats = self._benchmark_implementation(
                    "Hybrid", size, load_factor, num_searches)

                # Store results
                self.results.extend([traditional_stats, elastic_stats, hybrid_stats])

        return self._generate_report()

    def _benchmark_implementation(self, impl_type: str, size: int,
                                load_factor: float, num_searches: int) -> Dict:
        num_items = int(size * load_factor)

        # Initialize appropriate hash table
        if impl_type == "Traditional":
            table = TraditionalHashTable(size)
        elif impl_type == "Elastic":
            table = ElasticHashTable(size, 1-load_factor)
        else:  # Hybrid
            table = HybridHashTable(size)

        # Benchmark insertions
        insert_probes = []
        insert_start_time = time.time()

        for i in range(num_items):
            try:
                probes = table.insert(f"key{i}")
                insert_probes.append(probes)
            except Exception as e:
                print(f"Insertion failed at i={i}: {e}")
                break

        insert_time = time.time() - insert_start_time
        insert_stats = table._calculate_stats("insert", insert_probes, insert_time)

        # Benchmark searches
        search_probes = []
        search_start_time = time.time()

        for _ in range(num_searches):
            key = f"key{random.randint(0, num_items-1)}"
            probes = table.search(key)
            search_probes.append(probes)

        search_time = time.time() - search_start_time
        search_stats = table._calculate_stats("search", search_probes, search_time)

        return {
            "implementation": impl_type,
            "size": size,
            "load_factor": load_factor,
            "insert_stats": insert_stats,
            "search_stats": search_stats
        }

    def _generate_report(self) -> Dict:
        df = pd.DataFrame([
            {
                'Implementation': r['implementation'],
                'Size': r['size'],
                'Load Factor': r['load_factor'],
                'Operation': 'Insert',
                'Avg Probes': r['insert_stats'].avg_probes,
                'Max Probes': r['insert_stats'].max_probes,
                'Ops/Second': r['insert_stats'].operations_per_second
            } for r in self.results
        ] + [
            {
                'Implementation': r['implementation'],
                'Size': r['size'],
                'Load Factor': r['load_factor'],
                'Operation': 'Search',
                'Avg Probes': r['search_stats'].avg_probes,
                'Max Probes': r['search_stats'].max_probes,
                'Ops/Second': r['search_stats'].operations_per_second
            } for r in self.results
        ])

        return {
            'raw_data': df,
            'summary': self._generate_summary_plots(df)
        }

    def _generate_summary_plots(self, df: pd.DataFrame) -> Dict[str, plt.Figure]:
        plots = {}

        # Plot average probes vs load factor for each implementation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        for operation, ax in [('Insert', ax1), ('Search', ax2)]:
            data = df[df['Operation'] == operation]
            sns.lineplot(data=data, x='Load Factor', y='Avg Probes',
                        hue='Implementation', style='Implementation', ax=ax)
            ax.set_title(f'Average {operation} Probes vs Load Factor')
            ax.grid(True)

        plots['probes'] = fig

        # Plot operations per second
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        for operation, ax in [('Insert', ax1), ('Search', ax2)]:
            data = df[df['Operation'] == operation]
            sns.lineplot(data=data, x='Load Factor', y='Ops/Second',
                        hue='Implementation', style='Implementation', ax=ax)
            ax.set_title(f'{operation} Operations per Second vs Load Factor')
            ax.grid(True)

        plots['performance'] = fig

        return plots

def run_demo():
    # Example usage
    sizes = [1000, 10000, 100000, 1000000]
    load_factors = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]

    benchmark = HashTableBenchmark(sizes=sizes, load_factors=load_factors)
    results = benchmark.run_comprehensive_benchmark()

    # Save plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results['summary']['probes'].savefig(f'probe_comparison_{timestamp}.png')
    results['summary']['performance'].savefig(f'performance_comparison_{timestamp}.png')

    # Print summary statistics
    df = results['raw_data']
    print("\nSummary Statistics:")
    print(df.groupby(['Implementation', 'Operation'])[['Avg Probes', 'Ops/Second']].mean())

if __name__ == "__main__":
    run_demo()
