"""
Scaling benchmark: shows how PCIe round-trip cost grows with matrix size.
Tests to_cooh and diagonal_op_ at increasing sizes.
"""
import time, torch, statistics

WARMUP = 3
REPEATS = 30
SEED = 42


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
    return statistics.median(times)


def make_bsr(n_blocks, block_size, device, density=0.1, seed=SEED):
    gen = torch.Generator().manual_seed(seed)
    crow, cols, vals = [0], [], []
    nnz_per_row = max(1, int(n_blocks * density))
    for i in range(n_blocks):
        col_set = {i}
        while len(col_set) < nnz_per_row:
            col_set.add(torch.randint(0, n_blocks, (1,), generator=gen).item())
        sorted_cols = sorted(col_set)
        cols.extend(sorted_cols)
        crow.append(crow[-1] + len(sorted_cols))
        for _ in sorted_cols:
            vals.append(torch.randn(block_size, block_size, generator=gen))
    crow_t = torch.tensor(crow, dtype=torch.int32, device=device)
    col_t = torch.tensor(cols, dtype=torch.int32, device=device)
    val_t = torch.stack(vals).to(device)
    return torch.sparse_bsr_tensor(crow_t, col_t, val_t,
                                   size=(n_blocks * block_size, n_blocks * block_size))


def orig_to_cooh(input):
    crow = input.crow_indices()
    col = input.col_indices()
    bsr_vals = input.values()
    bs = bsr_vals.shape[-2:]
    dummy = torch.sparse_csr_tensor(crow_indices=crow.to('cpu'), col_indices=col.to('cpu'),
                                    values=torch.zeros(bsr_vals.shape[0], device='cpu'))
    indices = dummy.to_sparse(layout=torch.sparse_coo).coalesce().indices().to(input.device)
    return torch.sparse_coo_tensor(indices, bsr_vals,
        [input.shape[0]//bs[0], input.shape[1]//bs[1], bs[0], bs[1]])


def opt_to_cooh(input):
    crow = input.crow_indices()
    col = input.col_indices()
    bsr_vals = input.values()
    bs = bsr_vals.shape[-2:]
    dummy = torch.sparse_csr_tensor(crow_indices=crow, col_indices=col,
                                    values=torch.zeros(bsr_vals.shape[0], device=input.device))
    indices = dummy.to_sparse(layout=torch.sparse_coo).coalesce().indices()
    return torch.sparse_coo_tensor(indices, bsr_vals,
        [input.shape[0]//bs[0], input.shape[1]//bs[1], bs[0], bs[1]])


def orig_diagonal_op_(input):
    crow = input.crow_indices()
    col = input.col_indices()
    bsr_vals = input.values()
    dm = bsr_vals.shape[-2]
    dummy = torch.sparse_csr_tensor(crow_indices=crow.to('cpu'), col_indices=col.to('cpu'),
                                    values=torch.zeros(bsr_vals.shape[0], device='cpu'))
    indices = dummy.to_sparse(layout=torch.sparse_coo).coalesce().indices().to(input.device)
    diag_mask = (indices[0] == indices[1])
    diag_indices = diag_mask.nonzero().squeeze(-1)
    return bsr_vals.diagonal(dim1=-2, dim2=-1)[diag_indices].flatten()


def opt_diagonal_op_(input):
    crow = input.crow_indices()
    col = input.col_indices()
    bsr_vals = input.values()
    dm = bsr_vals.shape[-2]
    dummy = torch.sparse_csr_tensor(crow_indices=crow, col_indices=col,
                                    values=torch.zeros(bsr_vals.shape[0], device=input.device))
    indices = dummy.to_sparse(layout=torch.sparse_coo).coalesce().indices()
    diag_mask = (indices[0] == indices[1])
    diag_indices = diag_mask.nonzero().squeeze(-1)
    return bsr_vals.diagonal(dim1=-2, dim2=-1)[diag_indices].flatten()


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping scaling benchmark.")
        return

    print("=" * 85)
    print("  CUDA Scaling Benchmark: PCIe Round-trip Impact vs Matrix Size")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 85)

    sizes = [50, 100, 200, 500, 1000, 2000]
    block_size = 6

    # ── to_cooh ──
    print(f"\n  to_cooh (Fix #2):")
    print(f"  {'N blocks':>10s}  {'NNZ':>8s}  {'Original':>12s}  {'Optimized':>12s}  {'Speedup':>10s}")
    print("  " + "-" * 60)

    for n in sizes:
        bsr = make_bsr(n, block_size, 'cuda', density=0.1)
        nnz = bsr.values().shape[0]
        t_orig = timer(lambda: orig_to_cooh(bsr), sync_cuda=True)
        t_opt = timer(lambda: opt_to_cooh(bsr), sync_cuda=True)
        sp = t_orig / t_opt if t_opt > 0 else float('inf')
        print(f"  {n:10d}  {nnz:8d}  {t_orig:10.3f}ms  {t_opt:10.3f}ms  {sp:8.2f}x")

    # ── diagonal_op_ ──
    print(f"\n  diagonal_op_ (Fix #1):")
    print(f"  {'N blocks':>10s}  {'NNZ':>8s}  {'Original':>12s}  {'Optimized':>12s}  {'Speedup':>10s}")
    print("  " + "-" * 60)

    for n in sizes:
        bsr = make_bsr(n, block_size, 'cuda', density=0.1)
        nnz = bsr.values().shape[0]
        t_orig = timer(lambda: orig_diagonal_op_(bsr), sync_cuda=True)
        t_opt = timer(lambda: opt_diagonal_op_(bsr), sync_cuda=True)
        sp = t_orig / t_opt if t_opt > 0 else float('inf')
        print(f"  {n:10d}  {nnz:8d}  {t_orig:10.3f}ms  {t_opt:10.3f}ms  {sp:8.2f}x")

    print("\n" + "=" * 85)


if __name__ == '__main__':
    main()
