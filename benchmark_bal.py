"""
A/B benchmark for Bundle Adjustment on ladybug dataset.
Compares original (pre-optimization) vs optimized sparse ops implementations.
"""
import time
import torch
import pypose as pp

from ba_helpers import Reproj, least_square_error
from datapipes.bal_loader import get_problem
from bae.optim import LM
from bae.utils.pysolvers import PCG

# ═══════════════════════════════════════════════════════════════════════════════
# Original implementations (CPU round-trips, redundant coalesces)
# ═══════════════════════════════════════════════════════════════════════════════

def _orig_to_cooh(input):
    """Original: CPU round-trip for CSR→COO conversion."""
    crow_indices = input.crow_indices()
    col_indices = input.col_indices()
    bsr_values = input.values()
    block_shape = bsr_values.shape[-2:]
    # ORIGINAL: .to('cpu') round-trip
    dummy_csr = torch.sparse_csr_tensor(crow_indices=crow_indices.to('cpu'),
                                        col_indices=col_indices.to('cpu'),
                                        values=torch.zeros(bsr_values.shape[0], device='cpu'))
    dummy_coo = dummy_csr.to_sparse(layout=torch.sparse_coo).coalesce()
    indices = dummy_coo.indices().to(input.device)
    return torch.sparse_coo_tensor(indices, bsr_values,
        [input.shape[0] // block_shape[0], input.shape[1] // block_shape[1],
         block_shape[0], block_shape[1]])


def _orig_diagonal_op_(input, offset=0, op=None):
    """Original: CPU round-trip for diagonal extraction."""
    crow_indices = input.crow_indices()
    col_indices = input.col_indices()
    bsr_values = input.values()
    m, n = input.shape[-2], input.shape[-1]
    dm, dn = (bsr_values.shape[-2], bsr_values.shape[-1]) if bsr_values.ndim > 1 else (1, 1)
    sm, sn = m // dm, n // dn

    if dm == dn and offset == 0:
        # ORIGINAL: CPU round-trip
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


def _to_bsr_helper(input):
    dummy_coo = torch.sparse_coo_tensor(input.indices(),
        torch.zeros(input.values().shape[0], device=input.device), input.shape[:2])
    dummy_coo = dummy_coo.coalesce()
    dummy_csr = dummy_coo.to_sparse(layout=torch.sparse_csr)
    return torch.sparse_bsr_tensor(dummy_csr.crow_indices(), dummy_csr.col_indices(),
        input.values(), [input.shape[0] * input.shape[-2], input.shape[1] * input.shape[-1]])


def _orig_add_op(input1, input2):
    """Original: triple coalesce."""
    input1_cooh = _orig_to_cooh(input1)
    input2_cooh = _orig_to_cooh(input2)
    input1_cooh = input1_cooh.coalesce()  # redundant
    input2_cooh = input2_cooh.coalesce()  # redundant
    result = input1_cooh + input2_cooh
    result = result.coalesce()
    return _to_bsr_helper(result)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_ba(problem_name, dataset_name, n_iters=20, use_original=False, device='cuda', verbose=False):
    """Run BA and return (total_time, final_loss, per_iter_times)."""

    # Load dataset
    dataset = get_problem(problem_name, dataset_name, use_quat=True)
    data = {k: v.to(device) for k, v in dataset.items() if isinstance(v, torch.Tensor)}

    input_dict = {
        "points_2d": data['points_2d'],
        "camera_indices": data['camera_index_of_observations'],
        "point_indices": data['point_index_of_observations']
    }

    model = Reproj(data['camera_params'][:, :10].clone(),
                   data['points_3d'].clone()).to(device)
    strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
    solver = PCG(tol=1e-4, maxiter=250)
    optimizer = LM(model, strategy=strategy, solver=solver, reject=30)

    # Monkey-patch if using original implementations
    if use_original:
        import bae.sparse.py_ops as py_ops
        orig_diagonal = py_ops.diagonal_op_
        orig_to_cooh = py_ops.to_cooh
        orig_add = py_ops.add_op
        py_ops.diagonal_op_ = _orig_diagonal_op_
        py_ops.to_cooh = _orig_to_cooh
        py_ops.add_op = _orig_add_op

    init_loss = optimizer.model.loss(input_dict, None).item()

    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.perf_counter()
    iter_times = []

    for i in range(n_iters):
        t0 = time.perf_counter()
        loss = optimizer.step(input_dict)
        torch.cuda.synchronize() if device == 'cuda' else None
        t1 = time.perf_counter()
        iter_times.append(t1 - t0)
        if verbose:
            print(f"      Iter {i}: loss={loss.item():.1f}, time={t1-t0:.3f}s")

    torch.cuda.synchronize() if device == 'cuda' else None
    total_time = time.perf_counter() - start

    final_loss = least_square_error(
        model.pose, model.points_3d,
        data['camera_index_of_observations'],
        data['point_index_of_observations'],
        data['points_2d']
    ).item()

    # Restore original functions
    if use_original:
        py_ops.diagonal_op_ = orig_diagonal
        py_ops.to_cooh = orig_to_cooh
        py_ops.add_op = orig_add

    return {
        'init_loss': init_loss,
        'final_loss': final_loss,
        'total_time': total_time,
        'iter_times': iter_times,
        'avg_iter': sum(iter_times) / len(iter_times),
    }


def main():
    problems = [
        ("problem-52-64053-pre", "venice"),       # 52 cams, 64053 pts
        ("problem-245-198739-pre", "venice"),     # 245 cams, 198739 pts
        ("problem-1102-780462-pre", "venice"),    # 1102 cams, 780462 pts
    ]

    n_iters = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("  Bundle Adjustment Benchmark (Optimized Code)")
    print(f"  Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ""))
    print(f"  Iterations per run: {n_iters}")
    print("=" * 80)
    print("\n  Note: Running optimized code only. For A/B comparison of specific")
    print("  functions (diagonal_op_, to_cooh, etc.), see benchmark_perf.py")
    print("  and benchmark_scaling.py which test implementations directly.\n")

    for problem_name, dataset_name in problems:
        print(f"\n  Problem: {problem_name}")
        print("-" * 80)

        # Warmup run (discarded)
        print("    Warmup run...", end=" ", flush=True)
        _ = run_ba(problem_name, dataset_name, n_iters=2, use_original=False, device=device)
        print("done")

        # Timed run
        print("    Timed run...", end=" ", flush=True)
        r = run_ba(problem_name, dataset_name, n_iters=n_iters, use_original=False, device=device)
        print(f"done")

        print(f"\n    {'Metric':<25s}  {'Value':>15s}")
        print("    " + "-" * 45)
        print(f"    {'Total time':<25s}  {r['total_time']:>12.3f}s")
        print(f"    {'Avg iter time':<25s}  {r['avg_iter']*1000:>12.2f}ms")
        print(f"    {'Initial loss':<25s}  {r['init_loss']:>15.1f}")
        print(f"    {'Final loss':<25s}  {r['final_loss']:>15.4f}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
