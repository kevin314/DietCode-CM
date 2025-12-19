#!/usr/bin/env python3
"""
Demonstrate runtime dynamic dispatch using the two-stage cost model.

Shows how the two-stage approach makes dynamic dispatch efficient:
- Stage 1 (base cost) computed once per micro-kernel
- Stage 2 (adaptation) computed cheaply for each runtime size
"""

import pickle
import numpy as np
from math import ceil

class Dispatcher:
    """
    Runtime dispatcher using two-stage cost model.

    Key advantage over single-stage: Stage 1 cost computed once,
    Stage 2 adaptation computed cheaply for any size.
    """

    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.fMK_model = model_data['fMK_model']
            self.stage1_feature_names = model_data['stage1_feature_names']

        self.kernel_library = [
            (16, 16, 8),
            (32, 32, 16),
            (64, 64, 32),
            (128, 128, 32),
        ]

        print("Pre-computing Stage 1 costs (micro-kernel quality)...")
        self.base_costs = {}
        for tiles in self.kernel_library:
            base_cost = self._compute_base_cost(tiles)
            self.base_costs[tiles] = base_cost
            print(f"  tiles={tiles}: base_cost={base_cost*1e6:.2f} us")

        print("\n Stage 1 costs pre-computed.\n")

    def _compute_base_cost(self, tiles):
        """Stage 1: Compute base cost from micro-kernel features."""
        from train_two_stage_model import extract_microkernel_features

        config = {'tile_m': tiles[0], 'tile_n': tiles[1], 'tile_k': tiles[2]}
        mk_features = extract_microkernel_features(config)

        feature_vector = np.array([[mk_features[k] for k in self.stage1_feature_names]])
        return self.fMK_model.predict(feature_vector)[0]

    def _compute_adaptation(self, M, N, K, tiles):
        """
        Compute adaptation factor for arbitrary problem size.

        Training used M=tile_m, N=tile_n (1 block in each dim).
        For multiple blocks, we scale by the number of blocks needed.
        """
        tile_m, tile_n, tile_k = tiles

        # How many blocks needed?
        num_blocks_m = ceil(M / tile_m)
        num_blocks_n = ceil(N / tile_n)
        num_k_tiles = ceil(K / tile_k)

        # Training used 1 block in M,N dimensions, so scale by block count
        # This is the key: runtime scales linearly with # of blocks!
        block_scale = num_blocks_m * num_blocks_n * num_k_tiles

        # Get base adaptation (what training used for 1 block)
        # For tiles=(16,16,8) training had: adapt ≈ 256
        # This represents the "iterations per block"
        from train_two_stage_model import compute_adaptation_factors
        training_config = {'tile_m': tile_m, 'tile_n': tile_n, 'tile_k': tile_k}
        base_adapt = compute_adaptation_factors(training_config)

        # Scale by number of blocks needed for this problem size
        return base_adapt['combined_factor'] * block_scale

    def dispatch(self, M, N, K):
        """
        Pick best micro-kernel for runtime size (M, N, K).
        """
        best_tiles = None
        best_time = float('inf')

        for tiles in self.kernel_library:
            # Stage 1: Load pre-computed base cost
            base_cost = self.base_costs[tiles]

            # Stage 2: Compute adaptation for this size
            adaptation = self._compute_adaptation(M, N, K, tiles)

            predicted_time = base_cost * adaptation

            if predicted_time < best_time:
                best_time = predicted_time
                best_tiles = tiles

        return best_tiles, best_time

    def matmul_dynamic(self, M, N, K):
        """Execute dynamic matmul - picks best config at runtime."""
        tiles, predicted_time = self.dispatch(M, N, K)
        print(f"Size ({M:4d}, {N:4d}, {K:4d}) → tiles={tiles}, "
              f"predicted={predicted_time*1e6:.1f} us")
        return tiles


def main():
    print("Loading cost model...\n")

    dispatcher = Dispatcher('two_stage_cost_model.pkl')

    # Test on various runtime sizes
    test_sizes = [
        (64, 32, 128),      # Small
        (128, 64, 256),     # Medium
        (256, 128, 512),    # Large
        (512, 256, 1024),   # Very large
        (1024, 512, 2048),  # Huge
        (64, 64, 64),       # Square small
        (256, 256, 256),    # Square medium
        (512, 512, 512),    # Square large
    ]

    import time
    start = time.time()

    for M, N, K in test_sizes:
        dispatcher.matmul_dynamic(M, N, K)

    elapsed = (time.time() - start) * 1000

    print(f"\nDispatched {len(test_sizes)} different sizes in {elapsed:.2f} ms")
    print(f"  ({elapsed/len(test_sizes):.3f} ms per dispatch)")


if __name__ == '__main__':
    main()
