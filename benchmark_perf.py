"""
A/B Benchmark + Equivalence for BAE performance optimizations.

Compares original (pre-optimization) implementations against optimized ones
by including both versions inline, avoiding git state issues.
"""
import time, sys, os, torch, statistics, warnings

WARMUP = 5
REPEATS = 50
SEED = 42
CUDA_AVAILABLE = torch.cuda.is_available()


def timer(fn, warmup=WARMUP, repeats=REPEATS, sync_cuda=False):
    for _ in range(warmup):
        fn()
    if sync_cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        if sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return {"median": round(statistics.median(times), 4),
            "mean": round(statistics.mean(times), 4),
            "min": round(min(times), 4),
            "std": round(statistics.stdev(times), 4) if len(times) > 1 else 0.0}


def make_bsr(n_blocks, block_size, device, density=0.3, seed=SEED):
    gen = torch.Generator().manual_seed(seed)
    n = n_blocks
    nnz_per_row = max(1, int(n * density))
    crow, cols, vals = [0], [], []
    for i in range(n):
        col_set = {i}
        while len(col_set) < nnz_per_row:
            col_set.add(torch.randint(0, n, (1,), generator=gen).item())
        sorted_cols = sorted(col_set)
        cols.extend(sorted_cols)
        crow.append(crow[-1] + len(sorted_cols))
        for _ in sorted_cols:
            vals.append(torch.randn(block_size, block_size, generator=gen))
    crow_t = torch.tensor(crow, dtype=torch.int32, device=device)
    col_t = torch.tensor(cols, dtype=torch.int32, device=device)
    val_t = torch.stack(vals).to(device)
    return torch.sparse_bsr_tensor(crow_t, col_t, val_t,
                                   size=(n * block_size, n * block_size))


# ═══════════════════════════════════════════════════════════════════════════════
# ORIGINAL implementations (pre-optimization)
# ═══════════════════════════════════════════════════════════════════════════════

def _orig_diagonal_op_(input, offset=0, op=None):
    """Original: moves crow/col to CPU, creates CSR on CPU, converts to COO on CPU,
    then moves indices back to GPU. Causes 3+ PCIe transfers per call."""
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


def _orig_add_op(input1, input2):
    """Original: triple coalesce."""
    input1_cooh = _orig_to_cooh(input1)
    input2_cooh = _orig_to_cooh(input2)
    input1_cooh = input1_cooh.coalesce()  # redundant
    input2_cooh = input2_cooh.coalesce()  # redundant
    result = input1_cooh + input2_cooh
    result = result.coalesce()  # only this one is needed
    return _to_bsr_helper(result)


def _to_bsr_helper(input):
    """Shared helper for to_bsr (used by original add_op)."""
    dummy_coo = torch.sparse_coo_tensor(input.indices(),
        torch.zeros(input.values().shape[0], device=input.device), input.shape[:2])
    dummy_coo = dummy_coo.coalesce()
    dummy_csr = dummy_coo.to_sparse(layout=torch.sparse_csr)
    crow_indices = dummy_csr.crow_indices()
    col_indices = dummy_csr.col_indices()
    values = input.values()
    block_shape = input.shape[-2:]
    return torch.sparse_bsr_tensor(crow_indices, col_indices, values,
        [input.shape[0] * block_shape[0], input.shape[1] * block_shape[1]])


def _orig_eye_blocks(output_shape, device, dtype):
    """Original: repeat() for identity blocks."""
    block_dim = output_shape[-1]
    eye = torch.eye(block_dim, device=device, dtype=dtype)
    return eye.unsqueeze(0).repeat(output_shape[0], 1, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZED implementations (current code)
# ═══════════════════════════════════════════════════════════════════════════════

def _opt_diagonal_op_(input, offset=0, op=None):
    """Optimized: keeps all tensors on device, no CPU round-trip."""
    crow_indices = input.crow_indices()
    col_indices = input.col_indices()
    bsr_values = input.values()
    m, n = input.shape[-2], input.shape[-1]
    dm, dn = (bsr_values.shape[-2], bsr_values.shape[-1]) if bsr_values.ndim > 1 else (1, 1)
    sm, sn = m // dm, n // dn

    if dm == dn and offset == 0:
        # OPTIMIZED: stay on device
        dummy_val = torch.zeros(bsr_values.shape[0], device=input.device)
        dummy = torch.sparse_csr_tensor(crow_indices=crow_indices,
                                        col_indices=col_indices,
                                        values=dummy_val)
        dummy_coo = dummy.to_sparse(layout=torch.sparse_coo).coalesce()
        indices = dummy_coo.indices()

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


def _opt_to_cooh(input):
    """Optimized: no CPU round-trip."""
    crow_indices = input.crow_indices()
    col_indices = input.col_indices()
    bsr_values = input.values()
    block_shape = bsr_values.shape[-2:]

    # OPTIMIZED: stay on device
    dummy_csr = torch.sparse_csr_tensor(crow_indices=crow_indices,
                                        col_indices=col_indices,
                                        values=torch.zeros(bsr_values.shape[0], device=input.device))
    dummy_coo = dummy_csr.to_sparse(layout=torch.sparse_coo).coalesce()
    indices = dummy_coo.indices()

    return torch.sparse_coo_tensor(indices, bsr_values,
        [input.shape[0] // block_shape[0], input.shape[1] // block_shape[1],
         block_shape[0], block_shape[1]])


def _opt_add_op(input1, input2):
    """Optimized: single coalesce (to_cooh already returns coalesced)."""
    input1_cooh = _opt_to_cooh(input1)
    input2_cooh = _opt_to_cooh(input2)
    result = (input1_cooh + input2_cooh).coalesce()
    return _to_bsr_helper(result)


def _opt_eye_blocks(output_shape, device, dtype):
    """Optimized: expand() instead of repeat() — no copy."""
    block_dim = output_shape[-1]
    eye = torch.eye(block_dim, device=device, dtype=dtype)
    return eye.unsqueeze(0).expand(output_shape[0], -1, -1)


# ═══════════════════════════════════════════════════════════════════════════════
# Equivalence checks
# ═══════════════════════════════════════════════════════════════════════════════

def check_equiv_diagonal(device):
    bsr = make_bsr(50, 4, device, density=0.2, seed=123)
    r_orig = _orig_diagonal_op_(bsr.clone())
    r_opt = _opt_diagonal_op_(bsr.clone())
    return torch.allclose(r_orig, r_opt, atol=1e-12)


def check_equiv_to_cooh(device):
    bsr = make_bsr(30, 4, device, density=0.3, seed=124)
    c_orig = _orig_to_cooh(bsr).coalesce()
    c_opt = _opt_to_cooh(bsr).coalesce()
    # Compare via dense reconstruction
    bs = 4; n = 30
    def to_dense(cooh):
        d = torch.zeros(n*bs, n*bs, device=device)
        idx = cooh.indices(); vals = cooh.values()
        for k in range(idx.shape[1]):
            r, c = idx[0,k].item(), idx[1,k].item()
            d[r*bs:(r+1)*bs, c*bs:(c+1)*bs] = vals[k]
        return d
    return torch.allclose(to_dense(c_orig), to_dense(c_opt), atol=1e-12)


def check_equiv_add(device):
    bsr1 = make_bsr(30, 4, device, density=0.2, seed=125)
    bsr2 = make_bsr(30, 4, device, density=0.2, seed=126)
    r_orig = _orig_add_op(bsr1, bsr2).to_dense()
    r_opt = _opt_add_op(bsr1, bsr2).to_dense()
    return torch.allclose(r_orig, r_opt, atol=1e-12)


def check_equiv_eye(device):
    shape = torch.Size([500, 6])
    dtype = torch.float64
    r_orig = _orig_eye_blocks(shape, device, dtype)
    r_opt = _opt_eye_blocks(shape, device, dtype)
    return torch.allclose(r_orig, r_opt, atol=1e-12)


def check_equiv_jacobian(device):
    """Verify optimized Jacobian (via installed bae) matches dense jacrev reference."""
    from bae.autograd.graph import jacobian as sparse_jacobian
    from bae.autograd.function import TrackingTensor as Track
    from torch.func import jacrev

    torch.manual_seed(200)
    dtype = torch.float64
    num_a, num_b = 8, 10; n = 20; dim = 3
    A0 = torch.randn(num_a, dim, device=device, dtype=dtype, requires_grad=True)
    B0 = torch.randn(num_b, dim, device=device, dtype=dtype, requires_grad=True)
    obs = torch.randn(n, dim, device=device, dtype=dtype)
    idx_a = torch.randint(0, num_a, (n,), device=device, dtype=torch.int32)
    idx_b = torch.randint(0, num_b, (n,), device=device, dtype=torch.int32)
    sel = torch.arange(0, min(n, 15), device=device, dtype=torch.int32)

    A_track = torch.nn.Parameter(Track(A0))
    B_track = torch.nn.Parameter(Track(B0))
    out = (A_track[idx_a][sel] + B_track[idx_b][sel]) - obs[sel]
    J_sparse = sparse_jacobian(out, [A_track, B_track])

    def f(A, B):
        return (A[idx_a][sel] + B[idx_b][sel]) - obs[sel]
    JA, JB = jacrev(f, argnums=(0, 1))(A0, B0)
    JA_flat = JA.reshape(sel.shape[0] * dim, num_a * dim)
    JB_flat = JB.reshape(sel.shape[0] * dim, num_b * dim)

    ok_a = torch.allclose(J_sparse[0].to_dense(), JA_flat, atol=1e-10)
    ok_b = torch.allclose(J_sparse[1].to_dense(), JB_flat, atol=1e-10)
    return ok_a and ok_b


# ═══════════════════════════════════════════════════════════════════════════════
# Performance benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

def bench_pair(name, orig_fn, opt_fn, device, n_blocks=200, block_size=6,
               density=0.1, warmup=WARMUP, repeats=REPEATS):
    """Benchmark original vs optimized on identical input."""
    sync = (device == 'cuda')

    if name == 'diagonal_op_':
        bsr = make_bsr(n_blocks, block_size, device, density=density)
        r_orig = timer(lambda: orig_fn(bsr), warmup=warmup, repeats=repeats, sync_cuda=sync)
        r_opt = timer(lambda: opt_fn(bsr), warmup=warmup, repeats=repeats, sync_cuda=sync)

    elif name == 'to_cooh':
        bsr = make_bsr(n_blocks, block_size, device, density=density)
        r_orig = timer(lambda: orig_fn(bsr), warmup=warmup, repeats=repeats, sync_cuda=sync)
        r_opt = timer(lambda: opt_fn(bsr), warmup=warmup, repeats=repeats, sync_cuda=sync)

    elif name == 'add_op':
        bsr1 = make_bsr(n_blocks, block_size, device, density=density)
        bsr2 = make_bsr(n_blocks, block_size, device, density=density, seed=99)
        r_orig = timer(lambda: orig_fn(bsr1, bsr2), warmup=warmup, repeats=repeats, sync_cuda=sync)
        r_opt = timer(lambda: opt_fn(bsr1, bsr2), warmup=warmup, repeats=repeats, sync_cuda=sync)

    elif name == 'eye_blocks':
        shape = torch.Size([n_blocks, block_size])
        dtype = torch.float64
        r_orig = timer(lambda: orig_fn(shape, device, dtype), warmup=warmup, repeats=repeats, sync_cuda=sync)
        r_opt = timer(lambda: opt_fn(shape, device, dtype), warmup=warmup, repeats=repeats, sync_cuda=sync)

    return r_orig, r_opt


def bench_jacobian_full(device):
    """Benchmark the full Jacobian pipeline (uses installed optimized bae)."""
    from bae.autograd.graph import jacobian as sparse_jacobian
    from bae.autograd.function import TrackingTensor as Track

    torch.manual_seed(SEED)
    dtype = torch.float64
    num_a, num_b = 50, 200; n = 500; dim = 9
    A0 = torch.randn(num_a, dim, device=device, dtype=dtype, requires_grad=True)
    B0 = torch.randn(num_b, 3, device=device, dtype=dtype, requires_grad=True)
    obs = torch.randn(n, 3, device=device, dtype=dtype)
    idx_a = torch.randint(0, num_a, (n,), device=device, dtype=torch.int32)
    idx_b = torch.randint(0, num_b, (n,), device=device, dtype=torch.int32)
    sync = (device == 'cuda')

    def run():
        for p in [A0, B0]:
            if hasattr(p, 'jactrace'): delattr(p, 'jactrace')
            if hasattr(p, 'optrace'): delattr(p, 'optrace')
        At = torch.nn.Parameter(Track(A0))
        Bt = torch.nn.Parameter(Track(B0))
        out = (At[idx_a][..., :3] + Bt[idx_b]) - obs
        return sparse_jacobian(out, [At, Bt])

    return timer(run, warmup=3, repeats=20, sync_cuda=sync)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_speedup(orig_ms, opt_ms):
    if opt_ms <= 0:
        return "N/A"
    ratio = orig_ms / opt_ms
    if ratio >= 1.0:
        return f"\033[32m{ratio:.2f}x faster\033[0m"
    else:
        return f"\033[31m{1/ratio:.2f}x slower\033[0m"


def main():
    devices = ['cpu']
    if CUDA_AVAILABLE:
        devices.append('cuda')

    print("=" * 80)
    print("  BAE Performance Optimization Benchmark")
    print(f"  PyTorch {torch.__version__}, CUDA: {CUDA_AVAILABLE}")
    if CUDA_AVAILABLE:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Warmup: {WARMUP}, Repeats: {REPEATS}, Seed: {SEED}")
    print("=" * 80)

    # ── Equivalence Tests ──
    print("\n\033[1m[EQUIVALENCE TESTS]\033[0m")
    print("  Verifying optimized code produces identical results to original...\n")

    equiv_checks = [
        ("diagonal_op_ (Fix #1)", check_equiv_diagonal),
        ("to_cooh (Fix #2)", check_equiv_to_cooh),
        ("add_op (Fix #3)", check_equiv_add),
        ("eye_blocks (Fix #9)", check_equiv_eye),
        ("jacobian (Fixes #9,#11)", check_equiv_jacobian),
    ]
    all_pass = True
    for device in devices:
        for name, fn in equiv_checks:
            try:
                ok = fn(device)
                status = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
                if not ok: all_pass = False
            except Exception as e:
                status = f"\033[31mERROR: {e}\033[0m"
                all_pass = False
            print(f"    {name:35s} [{device.upper():4s}]  {status}")

    if all_pass:
        print(f"\n    \033[32mAll equivalence tests passed.\033[0m")
    else:
        print(f"\n    \033[31mSome equivalence tests FAILED!\033[0m")

    # ── Performance Comparison ──
    print(f"\n\033[1m[PERFORMANCE COMPARISON]\033[0m")
    print(f"  {'Benchmark':25s}  {'Device':6s}  {'Original':>12s}  {'Optimized':>12s}  {'Speedup':>16s}")
    print("  " + "-" * 75)

    bench_pairs = [
        ("diagonal_op_", _orig_diagonal_op_, _opt_diagonal_op_),
        ("to_cooh", _orig_to_cooh, _opt_to_cooh),
        ("add_op", _orig_add_op, _opt_add_op),
        ("eye_blocks", _orig_eye_blocks, _opt_eye_blocks),
    ]

    for device in devices:
        for name, orig_fn, opt_fn in bench_pairs:
            r_orig, r_opt = bench_pair(name, orig_fn, opt_fn, device)
            sp = fmt_speedup(r_orig['median'], r_opt['median'])
            print(f"  {name:25s}  {device.upper():6s}  "
                  f"{r_orig['median']:10.3f}ms  {r_opt['median']:10.3f}ms  {sp}")

        # Full jacobian pipeline (only optimized, since we can't run the original)
        r_jac = bench_jacobian_full(device)
        print(f"  {'jacobian (full pipeline)':25s}  {device.upper():6s}  "
              f"{'N/A':>10s}    {r_jac['median']:10.3f}ms  (optimized only)")

    print("\n" + "=" * 80)
    print("  Notes:")
    print("  - diagonal_op_, to_cooh: Fix #1/#2 eliminate CPU<->GPU PCIe round-trips")
    print("  - add_op: Fix #3 removes 2 redundant .coalesce() calls")
    print("  - eye_blocks: Fix #9 uses expand() (shared memory) vs repeat() (copy)")
    print("  - Jacobian: Fixes #9,#11 applied; shown as optimized-only baseline")
    print("=" * 80)


if __name__ == '__main__':
    main()
