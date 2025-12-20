### Step 0: Extract Microkernel
Go to https://github.com/ChinDanIllinois/cs521finalproject, and pull the latest version. Follow the instructions to build XLA from source, targeting CPU https://openxla.org/xla/build_from_source. Utilize bazel to build the target "//xla/tools:hlo-opt". Then, use the following command to generate the corresponding LLVM IR, which will be printed to stderr.
**Command:**
```bash
./bazel-bin/xla/tools/hlo-opt --platform=cpu --stage=llvm-before-optimizations {hlo_file.hlo}
```
The hlo_file.hlo we provide should produce the corresponding template LLVM IR.

### Step 1: Generate LLVM IR Variations

Takes the template LLVM IR and generates 48 variations with different tile sizes.

**Command:**
```bash
python generate_variations.py
```

### Step 2: Parse Features from LLVM IR

Extracts features from each LLVM IR file (FLOPs, memory ops, loop structure).

**Command:**
```bash
python parse_all_variations.py
```

### Step 3: Compile and Benchmark All Variations

1. Compiles each `.ll` file to object code
2. Links with C++ harness
3. Executes variations and measures runtime

**Command:**
```bash
python benchmark_all.py
```

### Step 4: Train Two-Stage Cost Model

1. Loads benchmark data
2. Extracts **Stage 1 features** (micro-kernel quality):
   - tile_m, tile_n, tile_k
   - work_per_tile, bytes_per_tile
   - operational_intensity, loop structure
3. Computes **Stage 2 factors** (size adaptation):
   - occupancy_ratio, padding_ratio
   - iteration_efficiency, k_efficiency
4. Trains Gradient Boosting model on Stage 1
5. Evaluates complete two-stage model
6. Saves trained model

**Command:**
```bash
python train_two_stage_model.py
```

## Test Dynamic Dispatch

Use the trained model for runtime kernel selection.

**Command:**
```bash
python demo.py
```
