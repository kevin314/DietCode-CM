#!/usr/bin/env python3
"""
Generate unrolled variations of all 48 base kernels.

For each base kernel, creates versions with different unroll factors.
This expands the search space from 48 to 240 kernels.
"""

import os
import json
import subprocess

def apply_unrolling(input_llvm, output_llvm, unroll_factor):
    """
    Apply loop unrolling using LLVM opt.

    """
    opt_path = '/usr/lib/llvm-18/bin/opt'

    if unroll_factor == 1:
        if input_llvm == output_llvm:
            return True

        with open(input_llvm, 'r') as f_in:
            content = f_in.read()
        with open(output_llvm, 'w') as f_out:
            f_out.write(content)
        return True

    cmd = [
        opt_path,
        '-passes=loop-unroll',
        f'-unroll-count={unroll_factor}',
        input_llvm,
        '-S',
        '-o', output_llvm
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error unrolling {input_llvm} with factor {unroll_factor}:")
        print(result.stderr)
        return False

    return True


def generate_unrolled_variations(base_configs_file='kernels/configs.json',
                                 unroll_factors=[1, 4, 16],
                                 output_dir='kernels'):
    """
    Generate unrolled variations for all base kernels.

    Args:
        base_configs_file: Path to base kernel configs
        unroll_factors: List of unroll factors to try
        output_dir: Directory for output files
    """

    print(f"\nLoading base configurations from {base_configs_file}...")
    with open(base_configs_file, 'r') as f:
        base_configs = json.load(f)

    print(f"Found {len(base_configs)} base kernels")
    print(f"Unroll factors: {unroll_factors}")
    print(f"Total kernels to generate: {len(base_configs) * len(unroll_factors)}")

    os.makedirs(output_dir, exist_ok=True)

    all_configs = []
    kernel_id = 0

    print("\nGenerating unrolled variations...")

    for base_idx, base_config in enumerate(base_configs):
        base_llvm = base_config['file']
        tile_m = base_config['tile_m']
        tile_n = base_config['tile_n']
        tile_k = base_config['tile_k']

        for unroll_factor in unroll_factors:
            # Keep base kernels (unroll=1)
            if unroll_factor == 1:
                output_llvm = base_llvm
            else:
                output_llvm = os.path.join(output_dir, f'microkernel_u{unroll_factor}_{base_idx}.ll')

            success = apply_unrolling(base_llvm, output_llvm, unroll_factor)

            if not success:
                print(f"  [{kernel_id}] FAILED: base={base_idx}, unroll={unroll_factor}")
                continue

            config = {
                'id': kernel_id,
                'file': output_llvm,
                'base_id': base_config['id'],
                'tile_m': tile_m,
                'tile_n': tile_n,
                'tile_k': tile_k,
                'unroll_factor': unroll_factor,
                'vector_width': 1
            }

            all_configs.append(config)
            kernel_id += 1

            if kernel_id % 50 == 0:
                print(f"  Generated {kernel_id}/{len(base_configs) * len(unroll_factors)} kernels...")

    print(f"\nSuccessfully generated {len(all_configs)} kernels")

    output_config_file = os.path.join(output_dir, 'configs_unrolled.json')
    with open(output_config_file, 'w') as f:
        json.dump(all_configs, indent=2, fp=f)

    print(f"Saved configurations to {output_config_file}")

    return all_configs


if __name__ == '__main__':
    generate_unrolled_variations()
