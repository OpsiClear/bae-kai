"""
A/B comparison for Venice BA: Original vs Optimized sparse ops.
Runs each version in a subprocess to ensure clean torch.library state.
"""
import subprocess
import sys
import json
import time

BENCHMARK_SCRIPT = '''
import sys
import time
import json
import torch
import pypose as pp

# Inject original implementations BEFORE importing bae (before torch.library registration)
USE_ORIGINAL = {use_original}

if USE_ORIGINAL:
    # Patch the module before it's imported
    import bae.sparse.py_ops as py_ops_module

    # Original implementations with CPU round-trips
    def _orig_diagonal_op_(input, offset=0, op=None):
        crow_indices = input.crow_indices()
        col_indices = input.col_indices()
        bsr_values = input.values()
        m, n = input.shape[-2], input.shape[-1]
        dm, dn = (bsr_values.shape[-2], bsr_values.shape[-1]) if bsr_values.ndim > 1 else (1, 1)
        sm, sn = m // dm, n // dn
        if dm == dn and offset == 0:
            dummy_val = torch.zeros(bsr_values.shape[0], device='cpu')
            dummy = torch.sparse_csr_tensor(crow_indices=crow_indices.to('cpu'),
                                            col_indices=col_indices.to('cpu'),
                                            values=dummy_val)
            dummy_coo = dummy.to_sparse(layout=torch.sparse_coo).coalesce()
            indices = dummy_coo.indices().to(input.device)
            diag_mask = (indices[0] == indices[1])
            diag_indices = diag_mask.nonzero().squeeze(-1)
            block_diags = bsr_values.diagonal(dim1=-2, dim2=-1) if bsr_values.ndim > 1 else bsr_values
            values = block_diags[diag_indices]
            n_diag_blocks = sm if sm < sn else sn
            if diag_indices.shape[-1] == n_diag_blocks:
                results = values
            else:
                results = torch.zeros((n_diag_blocks, dm), dtype=values.dtype, device=values.device)
                results[indices[0, diag_indices]] = values
            if bsr_values.ndim > 1:
                results = torch.flatten(results, start_dim=-2, end_dim=-1)
            if op is not None:
                results = op(results)
                block_diags[diag_indices] = results.view(n_diag_blocks, dm) if bsr_values.ndim > 1 else results
            return results

    def _orig_to_cooh(input):
        crow_indices = input.crow_indices()
        col_indices = input.col_indices()
        bsr_values = input.values()
        block_shape = bsr_values.shape[-2:]
        dummy_csr = torch.sparse_csr_tensor(crow_indices=crow_indices.to('cpu'),
                                            col_indices=col_indices.to('cpu'),
                                            values=torch.zeros(bsr_values.shape[0], device='cpu'))
        dummy_coo = dummy_csr.to_sparse(layout=torch.sparse_coo).coalesce()
        indices = dummy_coo.indices().to(input.device)
        return torch.sparse_coo_tensor(indices, bsr_values,
            [input.shape[0] // block_shape[0], input.shape[1] // block_shape[1],
             block_shape[0], block_shape[1]])

    def _orig_add_op(input1, input2):
        input1_cooh = _orig_to_cooh(input1)
        input2_cooh = _orig_to_cooh(input2)
        input1_cooh = input1_cooh.coalesce()
        input2_cooh = input2_cooh.coalesce()
        result = input1_cooh + input2_cooh
        result = result.coalesce()
        return py_ops_module.to_bsr(result)

    # Patch the functions
    py_ops_module.diagonal_op_ = _orig_diagonal_op_
    py_ops_module.to_cooh = _orig_to_cooh
    py_ops_module.add_op = _orig_add_op

# Now import the rest
from ba_helpers import Reproj, least_square_error
from datapipes.bal_loader import get_problem
from bae.optim import LM
from bae.utils.pysolvers import PCG

import warnings
warnings.filterwarnings('ignore')

def run_ba(problem_name, dataset_name, n_iters, device):
    dataset = get_problem(problem_name, dataset_name, use_quat=True)
    data = {{k: v.to(device) for k, v in dataset.items() if isinstance(v, torch.Tensor)}}

    input_dict = {{
        "points_2d": data['points_2d'],
        "camera_indices": data['camera_index_of_observations'],
        "point_indices": data['point_index_of_observations']
    }}

    model = Reproj(data['camera_params'][:, :10].clone(), data['points_3d'].clone()).to(device)
    strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
    solver = PCG(tol=1e-4, maxiter=250)
    optimizer = LM(model, strategy=strategy, solver=solver, reject=30)

    init_loss = optimizer.model.loss(input_dict, None).item()

    # Warmup
    for _ in range(2):
        optimizer.step(input_dict)

    # Reset model for timed run
    model = Reproj(data['camera_params'][:, :10].clone(), data['points_3d'].clone()).to(device)
    optimizer = LM(model, strategy=strategy, solver=solver, reject=30)

    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.perf_counter()

    for _ in range(n_iters):
        optimizer.step(input_dict)

    torch.cuda.synchronize() if device == 'cuda' else None
    total_time = time.perf_counter() - start

    final_loss = least_square_error(
        model.pose, model.points_3d,
        data['camera_index_of_observations'],
        data['point_index_of_observations'],
        data['points_2d']
    ).item()

    return {{'total_time': total_time, 'final_loss': final_loss, 'init_loss': init_loss}}

result = run_ba("{problem}", "{dataset}", {n_iters}, "{device}")
print(json.dumps(result))
'''

def run_benchmark(problem, dataset, n_iters, device, use_original):
    script = BENCHMARK_SCRIPT.format(
        use_original=use_original,
        problem=problem,
        dataset=dataset,
        n_iters=n_iters,
        device=device
    )

    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        cwd=r"C:\Users\opsiclear\Desktop\projects\colmap\bae"
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None

    # Parse the last line as JSON (skip warnings)
    lines = result.stdout.strip().split('\n')
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def main():
    problems = [
        ("problem-52-64053-pre", "venice"),
        ("problem-245-198739-pre", "venice"),
        ("problem-1102-780462-pre", "venice"),
    ]

    n_iters = 3
    device = 'cuda'

    print("=" * 85)
    print("  Bundle Adjustment A/B Comparison: Original vs Optimized")
    print("  (Each version runs in a separate subprocess for clean torch.library state)")
    print("=" * 85)

    for problem, dataset in problems:
        print(f"\n  Problem: {problem}")
        print("-" * 85)

        print("    Running OPTIMIZED...", end=" ", flush=True)
        r_opt = run_benchmark(problem, dataset, n_iters, device, use_original=False)
        if r_opt:
            print(f"done ({r_opt['total_time']:.2f}s)")
        else:
            print("FAILED")
            continue

        print("    Running ORIGINAL...", end=" ", flush=True)
        r_orig = run_benchmark(problem, dataset, n_iters, device, use_original=True)
        if r_orig:
            print(f"done ({r_orig['total_time']:.2f}s)")
        else:
            print("FAILED")
            continue

        speedup = r_orig['total_time'] / r_opt['total_time']

        print(f"\n    {'Metric':<20s}  {'Original':>12s}  {'Optimized':>12s}  {'Speedup':>12s}")
        print("    " + "-" * 60)
        print(f"    {'Total time':<20s}  {r_orig['total_time']:>10.2f}s   {r_opt['total_time']:>10.2f}s   {speedup:>10.2f}x")
        print(f"    {'Time/iter':<20s}  {r_orig['total_time']/n_iters*1000:>10.1f}ms  {r_opt['total_time']/n_iters*1000:>10.1f}ms  {speedup:>10.2f}x")
        print(f"    {'Final loss':<20s}  {r_orig['final_loss']:>12.4f}  {r_opt['final_loss']:>12.4f}")

    print("\n" + "=" * 85)


if __name__ == '__main__':
    main()
