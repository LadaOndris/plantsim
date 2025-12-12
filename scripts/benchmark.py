#!/usr/bin/env python3
"""
Benchmark script for plantsim simulation backends.

Runs benchmarks across different grid sizes and step counts,
compares backends, and generates performance plots.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --backends cpu cuda --grid-sizes 50 100 200
    python scripts/benchmark.py --help
"""

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_BACKENDS = ["cpu", "cuda"]
DEFAULT_GRID_SIZES = [20, 50, 100, 200, 400]
DEFAULT_STEP_COUNTS = [100, 500, 1000, 2000]
DEFAULT_BUILD_DIR = Path(__file__).parent.parent / "build"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "out"


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    backend: str
    grid_size: int
    steps: int
    duration_ms: int
    steps_per_second: float
    checksum: str
    timestamp: str


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    backends: list[str]
    grid_sizes: list[int]
    step_counts: list[int]
    build_dir: Path
    output_dir: Path
    runs_per_config: int = 3


# =============================================================================
# Git utilities
# =============================================================================

def get_git_commit_hash(short: bool = True) -> str:
    """Get the current git commit hash."""
    try:
        cmd = ["git", "rev-parse", "--short" if short else "", "HEAD"]
        cmd = [c for c in cmd if c]  # Remove empty strings
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_git_branch() -> str:
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


# =============================================================================
# Benchmark execution
# =============================================================================

def find_benchmark_executable(build_dir: Path, backend: str) -> Optional[Path]:
    """Find the benchmark executable for a given backend."""
    executable = build_dir / "bin" / f"bench_sim_{backend}"
    if executable.exists():
        return executable
    
    # Try alternative locations
    alt_path = build_dir / "bench" / "simulation" / f"bench_sim_{backend}"
    if alt_path.exists():
        return alt_path
    
    return None


def run_benchmark(
    executable: Path,
    steps: int,
    grid_size: int,
    backend: str,
) -> Optional[BenchmarkResult]:
    """Run a single benchmark and parse the output."""
    try:
        result = subprocess.run(
            [str(executable), str(steps), str(grid_size)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"  Error running benchmark: {result.stderr}", file=sys.stderr)
            return None
        
        # Parse output
        output = result.stdout
        duration_ms = 0
        steps_per_second = 0.0
        checksum = ""
        
        for line in output.split("\n"):
            if "completed in" in line.lower():
                # "Simulation completed in 123 ms"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "in" and i + 1 < len(parts):
                        try:
                            duration_ms = int(parts[i + 1])
                        except ValueError:
                            pass
            elif "steps/second" in line.lower():
                # "Steps/second: 1234.56"
                parts = line.split(":")
                if len(parts) >= 2:
                    try:
                        steps_per_second = float(parts[1].strip())
                    except ValueError:
                        pass
            elif "checksum" in line.lower():
                # "Map checksum: 0x1234567890abcdef"
                parts = line.split(":")
                if len(parts) >= 2:
                    checksum = parts[1].strip()
        
        return BenchmarkResult(
            backend=backend,
            grid_size=grid_size,
            steps=steps,
            duration_ms=duration_ms,
            steps_per_second=steps_per_second,
            checksum=checksum,
            timestamp=datetime.now().isoformat(),
        )
        
    except subprocess.TimeoutExpired:
        print(f"  Benchmark timed out", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return None


def run_all_benchmarks(config: BenchmarkConfig) -> list[BenchmarkResult]:
    """Run all benchmark configurations."""
    results: list[BenchmarkResult] = []
    
    # Find executables
    executables: dict[str, Path] = {}
    for backend in config.backends:
        exe = find_benchmark_executable(config.build_dir, backend)
        if exe:
            executables[backend] = exe
            print(f"Found {backend} executable: {exe}")
        else:
            print(f"Warning: Could not find executable for backend '{backend}'")
    
    if not executables:
        print("Error: No benchmark executables found!", file=sys.stderr)
        return results
    
    # Calculate total runs
    total_runs = (
        len(executables) 
        * len(config.grid_sizes) 
        * len(config.step_counts) 
        * config.runs_per_config
    )
    current_run = 0
    
    print(f"\nRunning {total_runs} benchmark configurations...")
    print("=" * 60)
    
    for backend, exe in executables.items():
        for grid_size in config.grid_sizes:
            for steps in config.step_counts:
                for run_idx in range(config.runs_per_config):
                    current_run += 1
                    print(
                        f"[{current_run}/{total_runs}] "
                        f"{backend} | grid={grid_size} | steps={steps} | run={run_idx + 1}"
                    )
                    
                    result = run_benchmark(exe, steps, grid_size, backend)
                    if result:
                        results.append(result)
                        print(
                            f"  -> {result.duration_ms}ms, "
                            f"{result.steps_per_second:.1f} steps/s"
                        )
    
    print("=" * 60)
    print(f"Completed {len(results)} successful benchmarks")
    
    return results


# =============================================================================
# Data export
# =============================================================================

def save_results_to_csv(
    results: list[BenchmarkResult],
    output_path: Path,
) -> None:
    """Save benchmark results to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "backend", "grid_size", "steps", "duration_ms", 
        "steps_per_second", "checksum", "timestamp"
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "backend": result.backend,
                "grid_size": result.grid_size,
                "steps": result.steps,
                "duration_ms": result.duration_ms,
                "steps_per_second": result.steps_per_second,
                "checksum": result.checksum,
                "timestamp": result.timestamp,
            })
    
    print(f"Results saved to: {output_path}")


# =============================================================================
# Plotting
# =============================================================================

def aggregate_results(
    results: list[BenchmarkResult],
) -> dict[str, dict[tuple[int, int], list[float]]]:
    """
    Aggregate results by backend, returning average steps/second.
    Returns: {backend: {(grid_size, steps): [steps_per_second, ...]}}
    """
    aggregated: dict[str, dict[tuple[int, int], list[float]]] = {}
    
    for result in results:
        if result.backend not in aggregated:
            aggregated[result.backend] = {}
        
        key = (result.grid_size, result.steps)
        if key not in aggregated[result.backend]:
            aggregated[result.backend][key] = []
        
        aggregated[result.backend][key].append(result.steps_per_second)
    
    return aggregated


def create_plots(
    results: list[BenchmarkResult],
    output_dir: Path,
    commit_hash: str,
) -> None:
    """Create and save benchmark plots."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plots")
        return
    
    if not results:
        print("No results to plot")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregated = aggregate_results(results)
    
    backends = list(aggregated.keys())
    grid_sizes = sorted(set(r.grid_size for r in results))
    step_counts = sorted(set(r.steps for r in results))
    
    # Color palette
    colors = plt.cm.tab10.colors
    backend_colors = {b: colors[i % len(colors)] for i, b in enumerate(backends)}
    
    # Plot 1: Steps/second vs Grid Size (for each step count)
    fig, axes = plt.subplots(
        1, len(step_counts), 
        figsize=(5 * len(step_counts), 5),
        sharey=True,
    )
    if len(step_counts) == 1:
        axes = [axes]
    
    for ax, steps in zip(axes, step_counts):
        for backend in backends:
            x_vals = []
            y_vals = []
            y_errs = []
            
            for grid_size in grid_sizes:
                key = (grid_size, steps)
                if key in aggregated[backend]:
                    values = aggregated[backend][key]
                    x_vals.append(grid_size)
                    y_vals.append(sum(values) / len(values))
                    y_errs.append(
                        (max(values) - min(values)) / 2 if len(values) > 1 else 0
                    )
            
            ax.errorbar(
                x_vals, y_vals, yerr=y_errs,
                marker="o", label=backend, color=backend_colors[backend],
                capsize=3, linewidth=2, markersize=6,
            )
        
        ax.set_xlabel("Grid Size")
        ax.set_title(f"Steps = {steps}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(grid_sizes)
        ax.set_xticklabels([str(g) for g in grid_sizes])
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[0].set_ylabel("Steps/Second")
    fig.suptitle(f"Performance vs Grid Size (commit: {commit_hash})")
    plt.tight_layout()
    
    plot_path = output_dir / "perf_vs_grid_size.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot: {plot_path}")
    
    # Plot 2: Steps/second vs Step Count (for each grid size)
    fig, axes = plt.subplots(
        1, len(grid_sizes),
        figsize=(5 * len(grid_sizes), 5),
        sharey=True,
    )
    if len(grid_sizes) == 1:
        axes = [axes]
    
    for ax, grid_size in zip(axes, grid_sizes):
        for backend in backends:
            x_vals = []
            y_vals = []
            y_errs = []
            
            for steps in step_counts:
                key = (grid_size, steps)
                if key in aggregated[backend]:
                    values = aggregated[backend][key]
                    x_vals.append(steps)
                    y_vals.append(sum(values) / len(values))
                    y_errs.append(
                        (max(values) - min(values)) / 2 if len(values) > 1 else 0
                    )
            
            ax.errorbar(
                x_vals, y_vals, yerr=y_errs,
                marker="o", label=backend, color=backend_colors[backend],
                capsize=3, linewidth=2, markersize=6,
            )
        
        ax.set_xlabel("Steps")
        ax.set_title(f"Grid = {grid_size}x{grid_size}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(step_counts)
        ax.set_xticklabels([str(s) for s in step_counts])
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[0].set_ylabel("Steps/Second")
    fig.suptitle(f"Performance vs Step Count (commit: {commit_hash})")
    plt.tight_layout()
    
    plot_path = output_dir / "perf_vs_steps.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot: {plot_path}")
    
    # Plot 3: Speedup comparison (if multiple backends)
    if len(backends) >= 2 and "cpu" in backends:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cpu_data = aggregated.get("cpu", {})
        
        # Get non-CPU backends for comparison
        compare_backends = [b for b in backends if b != "cpu"]
        n_backends = len(compare_backends)
        bar_width = 0.8 / max(n_backends, 1)
        
        for idx, backend in enumerate(compare_backends):
            backend_data = aggregated[backend]
            speedups = []
            labels = []
            
            for key in sorted(cpu_data.keys()):
                if key in backend_data:
                    cpu_avg = sum(cpu_data[key]) / len(cpu_data[key])
                    backend_avg = sum(backend_data[key]) / len(backend_data[key])
                    if cpu_avg > 0:
                        speedups.append(backend_avg / cpu_avg)
                        labels.append(f"{key[0]}x{key[0]}\n{key[1]} steps")
            
            # Center bars around integer positions
            offset = (idx - (n_backends - 1) / 2) * bar_width
            x_positions = [i + offset for i in range(len(speedups))]
            
            ax.bar(
                x_positions,
                speedups,
                width=bar_width,
                label=f"{backend} vs CPU",
                color=backend_colors[backend],
            )
        
        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Speedup (x times faster than CPU)")
        ax.set_title(f"Backend Speedup vs CPU (commit: {commit_hash})")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        
        plot_path = output_dir / "speedup_vs_cpu.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot: {plot_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark plantsim simulation backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--backends",
        nargs="+",
        default=DEFAULT_BACKENDS,
        help=f"Backends to benchmark (default: {DEFAULT_BACKENDS})",
    )
    parser.add_argument(
        "--grid-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_GRID_SIZES,
        help=f"Grid sizes to test (default: {DEFAULT_GRID_SIZES})",
    )
    parser.add_argument(
        "--step-counts",
        nargs="+",
        type=int,
        default=DEFAULT_STEP_COUNTS,
        help=f"Step counts to test (default: {DEFAULT_STEP_COUNTS})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per configuration (default: 3)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help=f"Build directory (default: {DEFAULT_BUILD_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Get git info for file naming
    commit_hash = get_git_commit_hash()
    branch = get_git_branch()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Plantsim Benchmark Runner")
    print(f"========================")
    print(f"Commit: {commit_hash} ({branch})")
    print(f"Backends: {args.backends}")
    print(f"Grid sizes: {args.grid_sizes}")
    print(f"Step counts: {args.step_counts}")
    print(f"Runs per config: {args.runs}")
    print()
    
    # Create configuration
    config = BenchmarkConfig(
        backends=args.backends,
        grid_sizes=args.grid_sizes,
        step_counts=args.step_counts,
        build_dir=args.build_dir,
        output_dir=args.output_dir,
        runs_per_config=args.runs,
    )
    
    # Run benchmarks
    results = run_all_benchmarks(config)
    
    if not results:
        print("No benchmark results collected!", file=sys.stderr)
        return 1
    
    # Save results
    data_dir = args.output_dir / "data"
    csv_filename = f"benchmark_{timestamp}_{commit_hash}.csv"
    csv_path = data_dir / csv_filename
    save_results_to_csv(results, csv_path)
    
    # Create plots
    if not args.no_plots:
        plots_dir = args.output_dir / "plots" / f"{timestamp}_{commit_hash}"
        create_plots(results, plots_dir, commit_hash)
    
    print()
    print("Benchmark complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
