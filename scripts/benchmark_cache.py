import time
import statistics
import pandas as pd
from core.cache_client import CacheClient
from data_sources.yahoo_finance import YahooFinanceSource
from utils.logging_utils import setup_logging
import logging

setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_cache_operations():
    cache = CacheClient()
    
    test_data = {
        'string': 'test_value',
        'dict': {'key': 'value', 'number': 42},
        'list': list(range(1000)),
        'dataframe': pd.DataFrame({'col1': range(100), 'col2': range(100, 200)})
    }
    
    results = {}
    
    for data_type, data in test_data.items():
        key = f"benchmark_{data_type}"
        
        write_times = []
        for _ in range(10):
            start = time.perf_counter()
            cache.set(key, data, ttl=60)
            write_times.append(time.perf_counter() - start)
        
        read_times = []
        for _ in range(10):
            start = time.perf_counter()
            cache.get(key)
            read_times.append(time.perf_counter() - start)
        
        results[data_type] = {
            'write_avg_ms': statistics.mean(write_times) * 1000,
            'write_std_ms': statistics.stdev(write_times) * 1000,
            'read_avg_ms': statistics.mean(read_times) * 1000,
            'read_std_ms': statistics.stdev(read_times) * 1000
        }
    
    return results

def benchmark_cache_hit_improvement():
    source = YahooFinanceSource()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    start = time.perf_counter()
    data1 = source.get_price_data(tickers, '2024-01-01', '2024-02-01')
    miss_time = time.perf_counter() - start

    start = time.perf_counter()
    data2 = source.get_price_data(tickers, '2024-01-01', '2024-02-01')
    hit_time = time.perf_counter() - start
    
    return {
        'cache_miss_seconds': miss_time,
        'cache_hit_seconds': hit_time,
        'speedup_factor': miss_time / hit_time if hit_time > 0 else float('inf')
    }

def main():
    print("\n" + "="*60)
    print("REDIS CACHE BENCHMARK")
    print("="*60)
    
    print("\n📊 Cache Operation Benchmarks (ms):")
    print("-"*50)
    results = benchmark_cache_operations()
    for data_type, metrics in results.items():
        print(f"\n{data_type.upper()}:")
        print(f"  Write: {metrics['write_avg_ms']:.3f}ms ± {metrics['write_std_ms']:.3f}ms")
        print(f"  Read:  {metrics['read_avg_ms']:.3f}ms ± {metrics['read_std_ms']:.3f}ms")
    
    print("\n" + "-"*50)
    print("\n Cache Hit Improvement (Price Data):")
    print("-"*50)
    hit_improvement = benchmark_cache_hit_improvement()
    print(f"Cache Miss: {hit_improvement['cache_miss_seconds']:.2f} seconds")
    print(f"Cache Hit:  {hit_improvement['cache_hit_seconds']:.2f} seconds")
    print(f"Speedup:    {hit_improvement['speedup_factor']:.1f}x faster")
    
    print("\n" + "="*60)
    print("Benchmark Complete")
    print("="*60)

if __name__ == "__main__":
    main()