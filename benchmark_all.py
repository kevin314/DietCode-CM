#!/usr/bin/env python3
import os
import json
import subprocess
import sys
from pathlib import Path

def compile_kernel(llvm_file, output_obj):
    cmd = ['clang', '-x', 'ir', '-c', '-O3', '-march=native', llvm_file, '-o', output_obj]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error compiling {llvm_file}:")
        print(result.stderr)
        return False
    return True

def link_benchmark(obj_file, cpp_file, output_exe):
    cmd = ['clang++', '-O3', '-march=native', '-o', output_exe, obj_file, cpp_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error linking {obj_file}:")
        print(result.stderr)
        return False
    return True

def run_benchmark(exe_file, tile_m, tile_n, tile_k, num_runs=100):
    result = subprocess.run([exe_file, '', str(num_runs), str(tile_m), str(tile_n), str(tile_k)],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {exe_file}:")
        print(result.stderr)
        return None

    output = result.stdout
    avg_time_us = None
    min_time_us = None
    max_time_us = None
    gflops = None

    for line in output.split('\n'):
        if 'Average time:' in line:
            # "Average time: 91.809 us (0.091809 ms)"
            avg_time_us = float(line.split()[2])
        elif 'Min time:' in line:
            min_time_us = float(line.split()[2])
        elif 'Max time:' in line:
            max_time_us = float(line.split()[2])
        elif 'Performance:' in line:
            gflops = float(line.split()[1])

    return {
        'avg_time_us': avg_time_us,
        'min_time_us': min_time_us,
        'max_time_us': max_time_us,
        'gflops': gflops
    }

def main():
    with open('kernels/parsed_features.json', 'r') as f:
        features = json.load(f)

    results = []

    print(f"Benchmarking {len(features)} kernels...\n")

    for i, config in enumerate(features):
        kernel_id = config['id']
        llvm_file = f'kernels/microkernel_{kernel_id}.ll'

        print(f"[{i+1}/{len(features)}] Processing {llvm_file} (tiles=[{config['tile_m']},{config['tile_n']},{config['tile_k']}])...", end=' ')

        # Compile to object
        os.makedirs('build', exist_ok=True)
        obj_file = f'build/microkernel_{kernel_id}.o'
        if not compile_kernel(llvm_file, obj_file):
            print("FAILED (compile)")
            continue

        # Link with benchmark harness
        exe_file = f'build/benchmark_{kernel_id}'
        if not link_benchmark(obj_file, 'benchmark_kernel.cpp', exe_file):
            print("FAILED (link)")
            continue

        # Run benchmark
        bench_results = run_benchmark(exe_file, config['tile_m'], config['tile_n'], config['tile_k'], num_runs=100)
        if bench_results is None:
            print("FAILED (run)")
            continue

        result = {**config, **bench_results}
        results.append(result)

        print(f"{bench_results['avg_time_us']:.2f} us ({bench_results['gflops']:.2f} GFLOPS)")

    output_file = 'kernels/benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, indent=2, fp=f)

    print(f"\n✓ Benchmarked {len(results)}/{len(features)} kernels")
    print(f"✓ Results saved to {output_file}")

    if results:
        times = [r['avg_time_us'] for r in results]
        print(f"\nSummary:")
        print(f"  Fastest: {min(times):.2f} us")
        print(f"  Slowest: {max(times):.2f} us")
        print(f"  Speedup: {max(times)/min(times):.2f}x")

if __name__ == '__main__':
    main()
