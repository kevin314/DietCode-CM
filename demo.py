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
import json

class Dispatcher:
    """
    Runtime dispatcher using two-stage cost model.

    Key advantage over single-stage: Stage 1 cost computed once,
    Stage 2 adaptation computed cheaply for any size.
    """

    def __init__(self, model_path, kernel_config_file='kernels/configs_unrolled.json'):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.fMK_model = model_data['fMK_model']
            self.stage1_feature_names = model_data['stage1_feature_names']

        with open(kernel_config_file, 'r') as f:
            configs = json.load(f)

        self.kernel_library = [
            {
                'tile_m': c['tile_m'],
                'tile_n': c['tile_n'],
                'tile_k': c['tile_k'],
                'unroll_factor': c.get('unroll_factor', 1),
                'file': c['file']
            }
            for c in configs
        ]

        print(f"Loaded {len(self.kernel_library)} kernel configurations from {kernel_config_file}")
        print("\nPre-computing Stage 1 costs (micro-kernel quality)...")

        self.base_costs = {}
        for i, config in enumerate(self.kernel_library):
            base_cost = self._compute_base_cost(config)
            config_key = (config['tile_m'], config['tile_n'], config['tile_k'], config['unroll_factor'])
            self.base_costs[config_key] = base_cost

        print(f"\nStage 1 costs pre-computed for {len(self.kernel_library)} configurations.\n")
    def _compute_base_cost(self, config):
        """Stage 1: Compute base cost from micro-kernel features."""
        from train_two_stage_model import extract_microkernel_features

        mk_features = extract_microkernel_features(config)

        feature_vector = np.array([[mk_features[k] for k in self.stage1_feature_names]])
        return self.fMK_model.predict(feature_vector)[0]

    def _compute_adaptation(self, M, N, K, config):
        """
        Compute adaptation factor for arbitrary problem size.

        Training used M=tile_m, N=tile_n (1 block in each dim).
        For multiple blocks, we scale by the number of blocks needed.
        """
        tile_m = config['tile_m']
        tile_n = config['tile_n']
        tile_k = config['tile_k']

        num_blocks_m = ceil(M / tile_m)
        num_blocks_n = ceil(N / tile_n)
        num_k_tiles = ceil(K / tile_k)

        block_scale = num_blocks_m * num_blocks_n * num_k_tiles

        from train_two_stage_model import compute_adaptation_factors
        training_config = {'tile_m': tile_m, 'tile_n': tile_n, 'tile_k': tile_k}
        base_adapt = compute_adaptation_factors(training_config)

        # Scale by number of blocks needed for this problem size
        return base_adapt['combined_factor'] * block_scale

    def dispatch(self, M, N, K):
        """
        Pick best micro-kernel for runtime size (M, N, K).
        """
        best_config = None
        best_time = float('inf')

        for config in self.kernel_library:
            # Stage 1: Load pre-computed base cost
            config_key = (config['tile_m'], config['tile_n'], config['tile_k'], config['unroll_factor'])
            base_cost = self.base_costs[config_key]

            # Stage 2: Compute adaptation for this size
            adaptation = self._compute_adaptation(M, N, K, config)

            predicted_time = base_cost * adaptation

            if predicted_time < best_time:
                best_time = predicted_time
                best_config = config

        return best_config, best_time

    def matmul_dynamic(self, M, N, K):
        """Execute dynamic matmul - picks best config at runtime."""
        config, predicted_time = self.dispatch(M, N, K)
        print(f"Size ({M:4d}, {N:4d}, {K:4d}) → tiles=({config['tile_m']},{config['tile_n']},{config['tile_k']}), "
              f"unroll={config['unroll_factor']}, predicted={predicted_time*1e6:.1f} us")
        return config


def main():
    print("Loading cost model...\n")

    dispatcher = Dispatcher('two_stage_cost_model.pkl')

    test_sizes = [
        (64, 32, 128),
        (128, 64, 256),
        (256, 128, 512),
        (512, 256, 1024),
        (1024, 512, 2048),
        (64, 64, 64),
        (256, 256, 256),
        (512, 512, 512),
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
