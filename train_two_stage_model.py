#!/usr/bin/env python3
"""
Train a DietCode-style two-stage cost model on real benchmark data.

Stage 1 (fMK): Learn micro-kernel quality (size-independent)
Stage 2 (fadapt): Compute adaptation factors (size-dependent ratios)
"""

import json
import numpy as np
from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pickle

# STAGE 1: Micro-kernel Features (Size-Independent)
def extract_microkernel_features(config):
    """
    Extract features that describe micro-kernel quality, independent of size.
    """
    tile_m = config['tile_m']
    tile_n = config['tile_n']
    tile_k = config['tile_k']

    features = {
        'tile_m': tile_m,
        'tile_n': tile_n,
        'tile_k': tile_k,
    }

    features['work_per_tile'] = tile_m * tile_n * tile_k * 2  # 2 FLOPs per inner iter
    features['bytes_per_tile'] = (tile_m * tile_k + tile_k * tile_n) * 4  # 4 bytes/float

    features['oi_microkernel'] = features['work_per_tile'] / features['bytes_per_tile']

    features['loop_depth'] = 5  # batch, M, N, K_outer, K_inner
    features['unroll_factor'] = config.get('unroll_factor', 1)
    features['vector_width'] = 1  # Currently no vectorization

    features['inner_loop_size'] = tile_k

    features['aspect_ratio_mn'] = tile_m / tile_n
    features['aspect_ratio_mk'] = tile_m / tile_k

    return features

# STAGE 2: Adaptation Factors (Size-Dependent Ratios)
def compute_adaptation_factors(config, num_cores=16):
    """
    Compute size-dependent adaptation factors (ratios in [0,1]).

    These describe how well the micro-kernel fits this specific problem size.
    """
    tile_m = config['tile_m']
    tile_n = config['tile_n']
    tile_k = config['tile_k']

    # Problem size
    batch = 8
    M = tile_m  # In our data, M varies with tile_m
    N = tile_n  # In our data, N varies with tile_n
    K = 16 * tile_k 

    # Core Occupancy Factor (DietCode's fOCC)
    num_blocks_m = ceil(M / tile_m)  # Usually 1 for our data
    num_blocks_n = ceil(N / tile_n)  # Usually 1 for our data
    num_blocks = batch * num_blocks_m * num_blocks_n

    num_waves = ceil(num_blocks / num_cores)
    occupancy_ratio = num_blocks / (num_waves * num_cores)

    # Padding Factor (DietCode's fpad)
    actual_work = M * N * K
    padded_M = num_blocks_m * tile_m
    padded_N = num_blocks_n * tile_n
    padded_work = padded_M * padded_N * K
    padding_ratio = actual_work / padded_work if padded_work > 0 else 1.0

    # Iteration Efficiency
    # Ratio of useful work to total loop iterations
    num_k_tiles = ceil(K / tile_k)
    total_iterations = batch * num_blocks_m * num_blocks_n * num_k_tiles * tile_k
    ideal_iterations = batch * M * N * K
    iteration_efficiency = ideal_iterations / total_iterations if total_iterations > 0 else 1.0

    # K-dimension Efficiency
    k_efficiency = (K % tile_k) / tile_k if tile_k > 0 else 1.0
    k_efficiency = 1.0 - k_efficiency

    return {
        'occupancy_ratio': occupancy_ratio,
        'padding_ratio': padding_ratio,
        'iteration_efficiency': iteration_efficiency,
        'k_efficiency': k_efficiency,
        'combined_factor': occupancy_ratio * padding_ratio * iteration_efficiency * k_efficiency
    }

def train_two_stage_model(data):
    """
    Train the two-stage cost model.

    Returns:
        fMK_model: Stage 1 model (micro-kernel cost)
        stage1_features: Feature names for stage 1
        stage2_features: Feature names for stage 2
    """

    print("\nExtracting features...")
    all_data = []

    for config in data:
        mk_features = extract_microkernel_features(config)

        adapt_factors = compute_adaptation_factors(config)

        actual_runtime = config['avg_time_us'] / 1e6

        all_data.append({
            'mk_features': mk_features,
            'adapt_factors': adapt_factors,
            'actual_runtime': actual_runtime,
            'config': config
        })

    print("\nPreparing Stage 1 training data...")

    stage1_feature_names = list(all_data[0]['mk_features'].keys())
    X_mk = np.array([[d['mk_features'][k] for k in stage1_feature_names] for d in all_data])

    y_mk = np.array([d['actual_runtime'] / d['adapt_factors']['combined_factor']
                     for d in all_data])

    print(f"Stage 1 features: {len(stage1_feature_names)}")
    print(f"Samples: {len(X_mk)}")

    # Split data
    indices = np.arange(len(X_mk))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_mk_train, X_mk_test = X_mk[train_idx], X_mk[test_idx]
    y_mk_train, y_mk_test = y_mk[train_idx], y_mk[test_idx]

    # Train Stage 1 model
    print("\nTraining Stage 1 model (fMK - micro-kernel cost)...")
    fMK_model = GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=4,
        random_state=42
    )
    fMK_model.fit(X_mk_train, y_mk_train)

    # Evaluate Stage 1
    y_mk_train_pred = fMK_model.predict(X_mk_train)
    y_mk_test_pred = fMK_model.predict(X_mk_test)

    print(f"  Train R²: {r2_score(y_mk_train, y_mk_train_pred):.6f}")
    print(f"  Test R²:  {r2_score(y_mk_test, y_mk_test_pred):.6f}")

    print("\n  Stage 1 Feature Importance:")
    importances = fMK_model.feature_importances_
    for name, imp in sorted(zip(stage1_feature_names, importances),
                           key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {name:25s}: {imp:.4f}")

    return fMK_model, all_data, stage1_feature_names, train_idx, test_idx


def evaluate_two_stage_model(fMK_model, all_data, stage1_feature_names,
                             train_idx, test_idx):
    predictions = []
    actuals = []
    base_costs = []
    adaptation_factors = []

    for i, d in enumerate(all_data):
        mk_feature_vector = np.array([[d['mk_features'][k] for k in stage1_feature_names]])
        base_cost = fMK_model.predict(mk_feature_vector)[0]

        adapt = d['adapt_factors']['combined_factor']

        predicted_runtime = base_cost * adapt

        predictions.append(predicted_runtime)
        actuals.append(d['actual_runtime'])
        base_costs.append(base_cost)
        adaptation_factors.append(adapt)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Split into train/test
    train_pred = predictions[train_idx]
    train_actual = actuals[train_idx]
    test_pred = predictions[test_idx]
    test_actual = actuals[test_idx]

    train_r2 = r2_score(train_actual, train_pred)
    test_r2 = r2_score(test_actual, test_pred)
    train_mae = mean_absolute_error(train_actual, train_pred)
    test_mae = mean_absolute_error(test_actual, test_pred)

    print(f"\nModel Performance:")
    print(f"  Train R²:  {train_r2:.6f}")
    print(f"  Test R²:   {test_r2:.6f}")
    print(f"  Train MAE: {train_mae*1e6:.2f} us")
    print(f"  Test MAE:  {test_mae*1e6:.2f} us")

    # Show test predictions
    print("\n" + "="*80)
    print("TEST SET PREDICTIONS")
    print("="*80)
    print(f"{'Actual (us)':>12} {'Base Cost (us)':>15} {'Adapt':>8} {'Predicted (us)':>15} {'Error %':>10}")
    print("-"*80)

    for idx in test_idx[:10]:
        actual = actuals[idx] * 1e6
        base = base_costs[idx] * 1e6
        adapt = adaptation_factors[idx]
        pred = predictions[idx] * 1e6
        error = abs(actual - pred) / actual * 100

        print(f"{actual:12.2f} {base:15.2f} {adapt:8.3f} {pred:15.2f} {error:10.2f}%")

    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'predictions': predictions,
        'actuals': actuals,
        'base_costs': base_costs,
        'adaptation_factors': adaptation_factors
    }


def analyze_adaptation_factors(all_data):
    # Group by tile size
    print("\nAdaptation factors by tile configuration:")
    print("-"*80)
    print(f"{'Tiles':15} {'Occupancy':>12} {'Padding':>10} {'Iteration':>12} {'Combined':>12}")
    print("-"*80)

    for d in all_data[:10]:
        config = d['config']
        adapt = d['adapt_factors']
        tiles = f"[{config['tile_m']},{config['tile_n']},{config['tile_k']}]"

        print(f"{tiles:15} {adapt['occupancy_ratio']:12.3f} {adapt['padding_ratio']:10.3f} "
              f"{adapt['iteration_efficiency']:12.3f} {adapt['combined_factor']:12.3f}")

    print("\nAdaptation Factor Statistics:")
    print("-"*40)

    factors = ['occupancy_ratio', 'padding_ratio', 'iteration_efficiency', 'combined_factor']
    for factor_name in factors:
        values = [d['adapt_factors'][factor_name] for d in all_data]
        print(f"\n{factor_name}:")
        print(f"  Min:  {min(values):.4f}")
        print(f"  Max:  {max(values):.4f}")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Std:  {np.std(values):.4f}")


def main():
    print("Loading benchmark data...")
    with open('kernels/benchmark_results.json', 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} benchmark results\n")

    fMK_model, all_data, stage1_feature_names, train_idx, test_idx = train_two_stage_model(data)

    results = evaluate_two_stage_model(fMK_model, all_data, stage1_feature_names,
                                      train_idx, test_idx)

    # Analyze adaptation factors
    analyze_adaptation_factors(all_data)

    # Save model
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)

    model_data = {
        'fMK_model': fMK_model,
        'stage1_feature_names': stage1_feature_names,
        'metadata': {
            'train_r2': results['train_r2'],
            'test_r2': results['test_r2'],
            'num_samples': len(data)
        }
    }

    with open('two_stage_cost_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("Saved to two_stage_cost_model.pkl")


if __name__ == '__main__':
    main()
