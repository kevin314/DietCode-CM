#!/usr/bin/env python3
"""
Parser to count FLOPs and memory operations from LLVM IR.

This extracts:
1. Number of FLOP instructions (fmul, fadd, etc.) in the innermost loop
2. Memory operations (load, store) and bytes transferred
3. Loop bounds from icmp instructions
4. Total FLOPs and memory traffic
"""

import re


def count_flops_from_llvm(llvm_file):
    """
    Count total FLOPs by analyzing loop bounds and FLOP instructions.
    """
    with open(llvm_file, 'r') as f:
        content = f.read()

    # Count FLOP instructions in the code
    flop_instructions = {
        'fmul': 0,
        'fadd': 0,
        'fsub': 0,
        'fdiv': 0, 
        'fma': 0,   # Fused multiply-add (counts as 2 FLOPs)
    }

    for line in content.split('\n'):
        line = line.strip()

        if '= fmul' in line:
            flop_instructions['fmul'] += 1
        if '= fadd' in line:
            flop_instructions['fadd'] += 1
        if '= fsub' in line:
            flop_instructions['fsub'] += 1
        if '= fdiv' in line:
            flop_instructions['fdiv'] += 1
        if '= fma' in line or 'llvm.fma' in line:
            flop_instructions['fma'] += 1

    print("FLOP instructions found in code:")
    for op, count in flop_instructions.items():
        if count > 0:
            print(f"  {op}: {count}")

    loop_bounds = parse_loop_bounds(content)

    memory_ops = count_memory_operations(content)

    # Calculate FLOPs per iteration
    flops_per_iter = (flop_instructions['fmul'] +
                     flop_instructions['fadd'] +
                     flop_instructions['fsub'] +
                     flop_instructions['fdiv'] +
                     flop_instructions['fma'] * 2)

    # Total iterations = product of all loop bounds
    total_iters = 1
    for loop_name, bound in loop_bounds.items():
        if loop_name != 'K_outer':  # K_outer is dynamic
            total_iters *= bound

    # Handle dynamic K dimension
    if 'K_outer' in loop_bounds:
        total_flops = None
    else:
        total_flops = flops_per_iter * total_iters

    return {
        'flops_per_iteration': flops_per_iter,
        'loop_bounds': loop_bounds,
        'total_iterations': total_iters,
        'total_flops': total_flops,
        'flop_instructions': flop_instructions,
        'memory_ops': memory_ops
    }


def count_memory_operations(content):
    """
    Count memory operations (load/store) and estimate bytes transferred.

    Returns:
        dict with load/store counts and byte estimates
    """
    loads = 0
    stores = 0
    load_types = []
    store_types = []

    lines = content.split('\n')

    for line in lines:
        line = line.strip()

        # Count loads: %var = load <type>, ptr %addr
        if '= load' in line:
            loads += 1
            # Extract data type to estimate bytes
            # %11 = load float, ptr %10, align 4
            match = re.search(r'load (\w+),', line)
            if match:
                data_type = match.group(1)
                load_types.append(data_type)

        # Count stores: store <type> %val, ptr %addr
        if 'store' in line and '=' not in line.split('store')[0]:
            stores += 1
            # Extract data type
            # store float %16, ptr %accum_address, align 4
            match = re.search(r'store (\w+)', line)
            if match:
                data_type = match.group(1)
                store_types.append(data_type)

    # Calculate bytes per operation
    type_sizes = {
        'float': 4,
        'double': 8,
        'i64': 8,
        'i32': 4,
        'i16': 2,
        'i8': 1,
    }

    load_bytes = sum(type_sizes.get(t, 4) for t in load_types)
    store_bytes = sum(type_sizes.get(t, 4) for t in store_types)

    loads_from_tensors = 0
    loads_from_accumulator = 0
    stores_to_accumulator = 0
    stores_to_output = 0

    in_reduction_loop = False
    in_outer_reduction_exit = False

    for i, line in enumerate(lines):
        line = line.strip()

        # Track which loop we're in
        if 'out.loop_body.reduction:' in line:
            in_reduction_loop = True
            in_outer_reduction_exit = False
        elif 'out.loop_exit.reduction:' in line:
            in_reduction_loop = False
        elif 'out.loop_exit.outer_reduction:' in line:
            in_outer_reduction_exit = True

        # Classify loads/stores based on context
        if '= load float' in line:
            if in_reduction_loop:
                if 'accum_address' in line:
                    loads_from_accumulator += 1
                else:
                    loads_from_tensors += 1

        if 'store float' in line:
            if in_reduction_loop and 'accum_address' in line:
                stores_to_accumulator += 1
            elif in_outer_reduction_exit:
                stores_to_output += 1

    return {
        'total_loads': loads,
        'total_stores': stores,
        'load_bytes_per_iter': load_bytes,
        'store_bytes_per_iter': store_bytes,
        'loads_from_tensors': loads_from_tensors,
        'loads_from_accumulator': loads_from_accumulator,
        'stores_to_accumulator': stores_to_accumulator,
        'stores_to_output': stores_to_output,
    }


def parse_loop_bounds(content):
    """
    Extract loop bounds from LLVM IR by finding icmp instructions.
    """
    loop_bounds = {}

    # Find all loop comparisons
    # icmp uge i64 %indvar, BOUND
    for line in content.split('\n'):
        # Look for loop exit conditions
        # %2 = icmp uge i64 %out.bdot.indvar, 8
        match = re.search(r'icmp uge i64 %(\S+), (\d+)', line)
        if match:
            var_name = match.group(1)
            bound = int(match.group(2))

            if 'bdot' in var_name:
                loop_bounds['batch'] = bound
            elif 'lhs.0' in var_name:
                loop_bounds['M'] = bound
            elif 'rhs.1' in var_name:
                loop_bounds['N'] = bound
            elif 'reduction' in var_name and 'outer' not in var_name:
                loop_bounds['K_inner'] = bound

        # Check for dynamic bounds
        # icmp uge i64 %out.indvar.outer_reduction, %num_tiles
        if 'icmp uge' in line and '%num_tiles' in line:
            loop_bounds['K_outer'] = 'dynamic'

    return loop_bounds