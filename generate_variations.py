#!/usr/bin/env python3
"""
Generate LLVM IR variations from microkernel_template.ll.

This creates many variations with different tile sizes by modifying
loop bounds in the original LLVM IR using string replacement.

Usage:
    python generate_variations.py
"""

import itertools
import os
import json


def apply_replacements(llvm_content, tile_m, tile_n, tile_k):
    """
    Apply tile size replacements to LLVM IR.

    Args:
        llvm_content: Original LLVM IR as string
        tile_m: New M tile size (rows)
        tile_n: New N tile size (columns)
        tile_k: New K tile size (reduction dimension)

    Returns:
        Modified LLVM IR string
    """
    modified = llvm_content

    total_k = 16 * tile_k

    # M loop bound
    # Original: icmp uge i64 %out.indvar.lhs.0, 64
    modified = modified.replace(
        'icmp uge i64 %out.indvar.lhs.0, 64',
        f'icmp uge i64 %out.indvar.lhs.0, {tile_m}'
    )

    # N loop bound
    # Original: icmp uge i64 %out.indvar.rhs.1, 32
    modified = modified.replace(
        'icmp uge i64 %out.indvar.rhs.1, 32',
        f'icmp uge i64 %out.indvar.rhs.1, {tile_n}'
    )

    # K inner loop bound
    # Original: icmp uge i64 %out.indvar.reduction, 8
    modified = modified.replace(
        'icmp uge i64 %out.indvar.reduction, 8',
        f'icmp uge i64 %out.indvar.reduction, {tile_k}'
    )

    # K stride multiplier for outer reduction loop
    # Original: %outer_pos = mul i64 %out.indvar.outer_reduction, 8
    modified = modified.replace(

        'mul i64 %out.indvar.outer_reduction, 8',

        f'mul i64 %out.indvar.outer_reduction, {tile_k}'

    )

    # Replace tensor dimension declarations

    # Array type for A matrix: [batch x M x K]
    # Original: [8 x [64 x [128 x float]]]
    modified = modified.replace(
        '[8 x [64 x [128 x float]]]',
        f'[8 x [{tile_m} x [{total_k} x float]]]'
    )

    # Subarray type for A: [M x K]
    # Original: [64 x [128 x float]]
    modified = modified.replace(
        '[64 x [128 x float]]',
        f'[{tile_m} x [{total_k} x float]]'
    )

    # Array type for B matrix: [batch x K x N]
    # Original: [8 x [128 x [32 x float]]]
    modified = modified.replace(
        '[8 x [128 x [32 x float]]]',
        f'[8 x [{total_k} x [{tile_n} x float]]]'
    )

    # Subarray type for B: [K x N]
    # Original: [128 x [32 x float]]
    modified = modified.replace(
        '[128 x [32 x float]]',
        f'[{total_k} x [{tile_n} x float]]'
    )

    # Array type for C matrix (output): [batch x M x N]
    # Original: [8 x [64 x [32 x float]]]
    modified = modified.replace(
        '[8 x [64 x [32 x float]]]',
        f'[8 x [{tile_m} x [{tile_n} x float]]]'
    )

    # Subarray type for C: [M x N]
    # Original: [64 x [32 x float]]
    modified = modified.replace(
        '[64 x [32 x float]]',
        f'[{tile_m} x [{tile_n} x float]]'
    )

    return modified


def generate_variations(template_file, output_dir='kernels'):
    """
    Generate LLVM IR variations from template.

    Args:
        template_file: Path to clean LLVM IR template (microkernel_template.ll)
        output_dir: Directory to save variations

    Returns:
        List of configuration dicts
    """

    # Read template
    if not os.path.exists(template_file):
        print(f"Template file not found: {template_file}")
        return []

    with open(template_file, 'r') as f:
        template_content = f.read()

    print(f"   Template size: {len(template_content):,} characters")

    os.makedirs(output_dir, exist_ok=True)
    print(f"   Output directory: {output_dir}/")

    tile_m_options = [16, 32, 64, 128]      # M dimension tiles
    tile_n_options = [16, 32, 64, 128]      # N dimension tiles
    tile_k_options = [8, 16, 32]            # K dimension tiles

    all_configs = list(itertools.product(
        tile_m_options,
        tile_n_options,
        tile_k_options
    ))

    print(f"\nSearch space:")
    print(f"   Tile M: {tile_m_options}")
    print(f"   Tile N: {tile_n_options}")
    print(f"   Tile K: {tile_k_options}")
    print(f"   Total configurations: {len(all_configs)}")

    print(f"\nGenerating variations...")
    print("-" * 70)

    variations = []

    for i, (tile_m, tile_n, tile_k) in enumerate(all_configs):
        modified_llvm = apply_replacements(
            template_content,
            tile_m,
            tile_n,
            tile_k
        )

        output_file = os.path.join(output_dir, f'microkernel_{i}.ll')
        with open(output_file, 'w') as f:
            f.write(modified_llvm)

        config = {
            'id': i,
            'file': output_file,
            'tile_m': tile_m,
            'tile_n': tile_n,
            'tile_k': tile_k,
            'unroll_factors': [1, 1, 1],  # Fixed for now (can't change via string replace)
            'vector_width': 1              # Fixed for now
        }
        variations.append(config)

        if (i + 1) % 10 == 0 or (i + 1) == len(all_configs):
            print(f"   Generated {i + 1:3d}/{len(all_configs)} variations...")

    print(f"\nSuccessfully generated {len(variations)} variations")

    return variations


def show_examples(variations):
    """Display example variations."""
    print("\n" + "="*70)
    print("EXAMPLE VARIATIONS")
    print("="*70)

    # Show first, middle, and last
    indices = [0, 1, 2, 3, len(variations)-1]

    for idx in indices:
        v = variations[idx]
        print(f"\nVariation {v['id']}:")
        print(f"  File:   {v['file']}")
        print(f"  Tiles:  M={v['tile_m']}, N={v['tile_n']}, K={v['tile_k']}")
        print(f"  Config: tiles=[{v['tile_m']}, {v['tile_n']}, {v['tile_k']}]")


def verify_variation(variation_file):
    """
    Quick verification that the variation looks valid.

    Checks that loop bounds were actually changed.
    """
    with open(variation_file, 'r') as f:
        content = f.read()

    # Check that we have the expected structures
    has_loop_headers = all([
        'out.bdot.loop_header:' in content,
        'out.loop_header.lhs.0:' in content,
        'out.loop_header.rhs.1:' in content,
        'out.loop_header.reduction:' in content,
    ])

    has_comparisons = 'icmp uge' in content
    has_arithmetic = 'fmul float' in content and 'fadd float' in content

    return has_loop_headers and has_comparisons and has_arithmetic


def main():
    template_file = 'microkernel_template.ll'
    output_dir = 'kernels'

    variations = generate_variations(template_file, output_dir)

    if not variations:
        print("\nNo variations generated. Exiting.")
        return

    show_examples(variations)

    import random
    test_indices = random.sample(range(len(variations)), min(5, len(variations)))

    print(f"\nVerifying {len(test_indices)} random variations...")
    all_valid = True

    for idx in test_indices:
        v = variations[idx]
        is_valid = verify_variation(v['file'])

        if not is_valid:
            all_valid = False

    if all_valid:
        print("\nAll tested variations look valid!")
    else:
        print("\nSome variations may have issues. Check manually.")

    config_file = os.path.join(output_dir, 'configs.json')
    with open(config_file, 'w') as f:
        json.dump(variations, f, indent=2)

    print(f"\nSaved config metadata to {config_file}")

if __name__ == '__main__':
    main()
