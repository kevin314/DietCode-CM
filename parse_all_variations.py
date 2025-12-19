#!/usr/bin/env python3
"""
Parse all generated LLVM IR variations and extract features.

This demonstrates that each variation has different features
even though they perform the same computation.
"""

import json
import os
from parse_llvm_flops import count_flops_from_llvm


def parse_all_variations(config_file='kernels/configs.json'):
    """Parse all generated variations and extract features."""
    with open(config_file, 'r') as f:
        configs = json.load(f)

    print(f"\nLoaded {len(configs)} configurations from {config_file}")

    print(f"\nParsing variations...")
    print("-" * 70)

    results = []

    for i, config in enumerate(configs):
        analysis = count_flops_from_llvm(config['file'])

        mem_ops = analysis['memory_ops']
        loop_bounds = analysis['loop_bounds']

        bytes_from_global = mem_ops['loads_from_tensors'] * 4
        oi_per_iter = analysis['flops_per_iteration'] / bytes_from_global if bytes_from_global > 0 else 0

        # Calculate total iterations (excluding K_outer which is dynamic)
        total_iters = analysis['total_iterations']

        K = 128  # Assume this for now
        k_outer_iters = K // config['tile_k']

        total_flops_static = total_iters * k_outer_iters * analysis['flops_per_iteration']
        total_bytes_static = total_iters * k_outer_iters * bytes_from_global
        oi_total = total_flops_static / total_bytes_static if total_bytes_static > 0 else 0

        result = {
            'id': config['id'],
            'tile_m': config['tile_m'],
            'tile_n': config['tile_n'],
            'tile_k': config['tile_k'],

            # Per iteration (innermost loop)
            'flops_per_iter': analysis['flops_per_iteration'],
            'loads_from_tensors': mem_ops['loads_from_tensors'],
            'bytes_per_iter': bytes_from_global,
            'oi_per_iter': oi_per_iter,

            # Total iterations
            'total_iterations': total_iters,
            'k_outer_iterations': k_outer_iters,

            # Total work (for K=128)
            'total_flops': total_flops_static,
            'total_bytes': total_bytes_static,
            'oi_total': oi_total,
        }

        results.append(result)

        if (i + 1) % 10 == 0 or (i + 1) == len(configs):
            print(f"   Parsed {i + 1:3d}/{len(configs)} variations...")

    print(f"\nSuccessfully parsed all {len(results)} variations")

    return results


def print_tile_features(results):
    print(f"\n{'ID':<4} {'Tiles':<20} {'Total Iters':<15} {'Total FLOPs':<15} {'OI_total':<10}")
    print("-" * 80)

    indices = [0, 12, 24, 36, 47]

    for idx in indices:
        r = results[idx]
        tiles_str = f"[{r['tile_m']}, {r['tile_n']}, {r['tile_k']}]"
        print(f"{r['id']:<4} {tiles_str:<20} {r['total_iterations']:<15,} "
              f"{r['total_flops']:<15,} {r['oi_total']:<10.2f}")


    print(f"\nTile configurations:")
    tile_m_values = sorted(set(r['tile_m'] for r in results))
    tile_n_values = sorted(set(r['tile_n'] for r in results))
    tile_k_values = sorted(set(r['tile_k'] for r in results))

    print(f"  Tile M options: {tile_m_values}")
    print(f"  Tile N options: {tile_n_values}")
    print(f"  Tile K options: {tile_k_values}")


def main():
    results = parse_all_variations()
    print_tile_features(results)

    output_file = 'kernels/parsed_features.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved parsed features to {output_file}")

if __name__ == '__main__':
    main()
