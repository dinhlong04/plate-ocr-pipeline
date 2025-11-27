"""
Benchmark Utilities for License Plate OCR Pipeline

Đo lường hiệu năng: latency, FPS, RAM.
"""

import gc
import time
import json
import csv
import numpy as np
from typing import Any, Dict, List, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class BenchmarkResult:
    """Kết quả benchmark cho một batch size."""
    batch_size: int
    n_runs: int
    n_images: int
    latency_avg: float = 0.0
    latency_std: float = 0.0
    fps: float = 0.0
    ram_peak: float = 0.0


@dataclass 
class BenchmarkReport:
    """Báo cáo benchmark tổng hợp."""
    model_path: str
    device: str
    preprocessing_steps: List[str]
    timestamp: str = ""
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "model_path": self.model_path,
            "device": self.device,
            "preprocessing_steps": self.preprocessing_steps,
            "timestamp": self.timestamp,
            "results": [asdict(r) for r in self.results]
        }


class Benchmarker:
    """
    Benchmark utility cho OCR Pipeline.
    
    Example:
        from utils.benchmark_utils import Benchmarker
        
        benchmarker = Benchmarker(pipeline)
        report = benchmarker.run(images, batch_sizes=[1, 2, 4, 8])
        benchmarker.export(report, "benchmark.csv")
    """
    
    def __init__(self, pipeline):
        """
        Args:
            pipeline: LicensePlateOCRPipeline instance
        """
        self.pipeline = pipeline
    
    @staticmethod
    def get_memory_mb() -> float:
        """Lấy RAM usage hiện tại (MB)."""
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    
    def benchmark_single_batch(
        self,
        images: List[Any],
        batch_size: int,
        n_runs: int = 50,
        warmup_runs: int = 5
    ) -> BenchmarkResult:
        """Benchmark một batch size cụ thể."""
        
        # Ensure model loaded
        if self.pipeline.model is None:
            self.pipeline.load_model()
        
        # Prepare images
        if len(images) < batch_size:
            images = (images * (batch_size // len(images) + 1))[:batch_size]
        else:
            images = images[:batch_size]
        
        gc.collect()
        
        # Preprocess once
        preprocessed = self.pipeline.preprocess(images)
        
        # Save original batch size
        old_batch_size = self.pipeline.batch_size
        self.pipeline.batch_size = batch_size
        
        # Warmup
        for _ in range(warmup_runs):
            self.pipeline.infer(preprocessed)
        
        gc.collect()
        
        # Benchmark
        latencies = []
        ram_samples = [self.get_memory_mb()]
        
        for _ in range(n_runs):
            start = time.perf_counter()
            self.pipeline.infer(preprocessed)
            latencies.append((time.perf_counter() - start) * 1000)
            ram_samples.append(self.get_memory_mb())
        
        # Restore batch size
        self.pipeline.batch_size = old_batch_size
        
        # Calculate stats
        latencies = np.array(latencies)
        avg_latency = float(np.mean(latencies))
        
        return BenchmarkResult(
            batch_size=batch_size,
            n_runs=n_runs,
            n_images=batch_size,
            latency_avg=avg_latency,
            latency_std=float(np.std(latencies)),
            fps=batch_size * 1000.0 / avg_latency,
            ram_peak=max(ram_samples)
        )
    
    def run(
        self,
        images: List[Any],
        batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
        n_runs: int = 50,
        warmup_runs: int = 5,
        verbose: bool = True
    ) -> BenchmarkReport:
        """
        Benchmark nhiều batch sizes.
        
        Args:
            images: List ảnh test
            batch_sizes: Danh sách batch sizes
            n_runs: Số lần chạy mỗi batch
            warmup_runs: Số lần warmup
            verbose: In kết quả
            
        Returns:
            BenchmarkReport
        """
        report = BenchmarkReport(
            model_path=self.pipeline.model_path,
            device=self.pipeline.device,
            preprocessing_steps=self.pipeline.preprocessor.get_enabled_steps(),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        if verbose:
            print("=" * 70)
            print("BENCHMARK: Batch Size Comparison")
            print("=" * 70)
            print(f"{'Batch':<10} {'Latency (ms)':<20} {'FPS':<15} {'RAM Peak':<15}")
            print(f"{'Size':<10} {'avg ± std':<20} {'img/s':<15} {'(MB)':<15}")
            print("-" * 70)
        
        for batch_size in batch_sizes:
            gc.collect()
            result = self.benchmark_single_batch(images, batch_size, n_runs, warmup_runs)
            report.results.append(result)
            
            if verbose:
                print(f"{batch_size:<10} "
                      f"{result.latency_avg:.2f} ± {result.latency_std:.2f}".ljust(20) + " "
                      f"{result.fps:<15.2f} "
                      f"{result.ram_peak:<15.2f}")
        
        if verbose:
            print("=" * 70)
            best_fps = max(report.results, key=lambda x: x.fps)
            best_latency = min(report.results, key=lambda x: x.latency_avg)
            print(f"📊 Best FPS: batch_size={best_fps.batch_size} ({best_fps.fps:.2f} img/s)")
            print(f"📊 Best Latency: batch_size={best_latency.batch_size} ({best_latency.latency_avg:.2f} ms)")
        
        return report
    
    def export(self, report: BenchmarkReport, output_path: str, format: str = "csv") -> str:
        """
        Xuất báo cáo ra file.
        
        Args:
            report: BenchmarkReport
            output_path: Đường dẫn file
            format: "csv" hoặc "json"
        """
        format = format.lower()
        
        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        elif format == "csv":
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["batch_size", "latency_avg_ms", "latency_std_ms", "fps", "ram_peak_mb"])
                for r in report.results:
                    writer.writerow([r.batch_size, f"{r.latency_avg:.4f}", 
                                    f"{r.latency_std:.4f}", f"{r.fps:.4f}", f"{r.ram_peak:.4f}"])
        else:
            raise ValueError(f"Format không hỗ trợ: {format}")
        
        print(f"✅ Report saved: {output_path}")
        return output_path