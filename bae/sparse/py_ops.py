"""Sparse matrix operations with Python and Triton implementations.

This module provides:
- jtj_diag: Efficient J^T @ J diagonal computation
- diagonal_op_: In-place diagonal extraction/modification
- spdiags_: Create sparse diagonal matrices
- bsr2bsc / bsc2bsr: Convert between BSR and BSC formats
- bsr_hcat: Horizontally concatenate BSR matrices
- sparse_bsr_bsc_mm / is_bsr_bsc_mm_available: BSR×BSC multiplication
- add_op: Sparse block matrix addition
- to_cooh / to_bsr: Format conversion utilities
"""

import logging
import warnings
from collections.abc import Callable

import torch
from torch.library import Library

from torch.utils._triton import has_triton
from ..exceptions import BAESparsityError

import os as _os
_skip_ext = _os.environ.get("BAE_SKIP_EXTENSIONS", "0") in ("1", "true", "yes")

if not _skip_ext:
    from .spgemm import convert_indices_from_csr_to_coo
else:
    # Fallback: spgemm just wraps at::_convert_indices_from_csr_to_coo
    convert_indices_from_csr_to_coo = torch._convert_indices_from_csr_to_coo

_logger = logging.getLogger(__name__)

# Check if triton is available (triton-windows package provides Windows support)
USE_TRITON = has_triton()

if not USE_TRITON:
    _logger.info("Triton not available. Using CPU fallback for diagonal operations.")
else:
    import triton
    from triton import language as tl

    @triton.jit
    def add_kernel(
        crow,
        col,
        out_ptr,
        nnz,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        js = block_start + tl.arange(0, BLOCK_SIZE)
        mask = js < nnz
        colj = tl.load(col + js, mask=mask)
        output = (tl.load(crow+colj) <= js) & (tl.load(crow+colj+1) > js)
        tl.store(out_ptr + js, output, mask=mask)

    def diagonal_op_triton_(x, op: Callable | None=None):
        nnz = x.col_indices().shape[-1]
        crow = x.crow_indices()
        col = x.col_indices()
        diag_mask = torch.zeros_like(col)
        grid = lambda meta: (triton.cdiv(nnz, meta["BLOCK_SIZE"]),)
        add_kernel[grid](crow, col, diag_mask, nnz, BLOCK_SIZE=4)
        
        return diag_mask
    @triton.jit
    def decompress_crow_kernel(
        crow_ptr,  # Pointer to the compressed row pointers
        row_indices_ptr,  # Pointer to the output row indices
        num_rows: tl.constexpr,  # Number of rows in the matrix
        num_nonzeros,  # Number of non-zero elements in the matrix
        BLOCK_SIZE: tl.constexpr,  # Block size for parallelism
    ):
        # Get the program ID
        pid = tl.program_id(0)
        
        # Compute the range of non-zero elements this block will handle
        block_start = pid * BLOCK_SIZE
        block_end = min(block_start + BLOCK_SIZE, num_nonzeros)
        
        j = 0
        # Loop over the non-zero elements in the current block
        for i in range(block_start, block_end):
            while j < num_rows:
                if i >= tl.load(crow_ptr + j) and i < tl.load(crow_ptr + j + 1):
                    tl.store(row_indices_ptr + i, j)
                else:
                    j += 1
        
    @triton.jit
    def aggr_kernel(
        crow,
        source,
        out_ptr,
        BLOCK_SIZE: "tl.constexpr",
    ):
        row = tl.program_id(axis=0)
        row_start = tl.load(crow + row)
        row_end = tl.load(crow + row + 1)
        js = tl.arange(row_start, row_end)
        valj = tl.load(source + js)
        output = tl.sum(valj)
        tl.store(out_ptr + row, output)
def jtj_diag(Jt):
    """Compute the diagonal of J^T @ J efficiently.

    Computes the diagonal elements of the normal equations matrix without
    forming the full J^T @ J product, which is more memory efficient.

    Parameters
    ----------
    Jt : torch.Tensor
        Transpose of the Jacobian matrix in BSR format.

    Returns
    -------
    torch.Tensor
        1D tensor containing the diagonal elements of J^T @ J.

    Notes
    -----
    The diagonal of J^T @ J equals the sum of squared elements in each
    column of J, which equals the sum of squared elements in each row of J^T.
    This is computed by squaring and summing the block values.
    """
    block_shape = Jt.values().shape[-2:]
    col_indices = Jt.col_indices()
    nnz = col_indices.shape[-1]
    crow_indices = Jt.crow_indices()
    srows = crow_indices.shape[-1] - 1
    values = Jt.values()
    values = values.flatten(start_dim=0, end_dim=1)
    dotp = torch.linalg.vecdot(values, values)
    # Guard contiguous() to avoid unnecessary copies if already contiguous
    crow_idx = crow_indices if crow_indices.is_contiguous() else crow_indices.contiguous()
    col_idx = col_indices if col_indices.is_contiguous() else col_indices.contiguous()
    cooa = convert_indices_from_csr_to_coo(crow_idx, col_idx, crow_indices.dtype == torch.int32, False)
    row_indices = cooa[0]
    if block_shape[0] == 1:
        ...
    else:
        row_indices = row_indices * block_shape[0]
        row_indices = row_indices.unsqueeze(-1)
        offsets = torch.arange(0, block_shape[0], device=row_indices.device, dtype=row_indices.dtype).unsqueeze(0)
        row_indices = row_indices + offsets
        row_indices = row_indices.flatten()
    diag_values = torch.zeros(srows * block_shape[0], device=values.device, dtype=values.dtype)
    diag_values.scatter_add_(0, row_indices.to(torch.int64), dotp)
    return diag_values

def diagonal_op_(input, offset: int=0, op: Callable | None=None):
    """Extract or modify the diagonal of a sparse block matrix.

    For CSR/BSR matrices, extracts diagonal elements. If an operation is
    provided, applies it in-place to the diagonal values.

    Parameters
    ----------
    input : torch.Tensor
        Sparse CSR or BSR matrix.
    offset : int, optional
        Diagonal offset (0 = main diagonal). Default is 0.
        Note: Only offset=0 is currently supported.
    op : callable, optional
        In-place operation to apply to diagonal values.
        Example: partial(torch.clamp_, min=1e-6)

    Returns
    -------
    torch.Tensor
        1D tensor containing diagonal values.

    Raises
    ------
    BAESparsityError
        If blocks are non-square or offset != 0.

    Examples
    --------
    >>> # Extract diagonal
    >>> diag = diagonal_op_(A)
    >>> # Clamp diagonal values
    >>> diagonal_op_(A, op=partial(torch.clamp_, min=1e-6))
    """
    crow_indices = input.crow_indices() # b + 1 dimensional
    col_indices = input.col_indices() # b + 1 dimensional
    bsr_values = input.values() # 1 + 2 dimensional
    m, n = input.shape[-2], input.shape[-1]
    if bsr_values.ndim > 1:
        dm, dn = (bsr_values.shape[-2], bsr_values.shape[-1])
    else:
        dm, dn = 1, 1
    sm, sn = m // dm, n // dn

    #simple case(block is square and offset is 0)
    if dm == dn and offset == 0:
        # Use Triton when available AND tensor is on CUDA (avoids CPU round-trip)
        use_triton_path = USE_TRITON and input.is_cuda

        if use_triton_path:
            # Fast path: use Triton kernel to compute diagonal mask on GPU
            diag_mask = diagonal_op_triton_(input)
            indices = None  # Defer indices computation - may not be needed
        else:
            # Fallback: CPU round-trip for COO conversion
            dummy_val = torch.zeros(bsr_values.shape[0], device='cpu')
            dummy = torch.sparse_csr_tensor(crow_indices=crow_indices.to('cpu'),
                                            col_indices=col_indices.to('cpu'),
                                            values=dummy_val)
            dummy_coo = dummy.to_sparse(layout=torch.sparse_coo).coalesce()
            indices = dummy_coo.indices().to(input.device)
            diag_mask = (indices[0] == indices[1])

        diag_indices = diag_mask.nonzero().squeeze(-1)
        if bsr_values.ndim > 1:
            block_diags = bsr_values.diagonal(dim1=-2, dim2=-1)
        else:
            block_diags = bsr_values
        values = block_diags[diag_indices]
        n_diag_blocks = sm if sm < sn else sn
        n_found = diag_indices.shape[-1]
        if n_found == n_diag_blocks:
            results = values
            # apply the inplace op
            if op is not None:
                results = op(results)
                block_diags[diag_indices] = results.view(n_diag_blocks, dm) if bsr_values.ndim > 1 else results
        else:
            # Slow path: some diagonal blocks are structurally absent
            if indices is None:
                # Compute indices on-demand (only when we need the slow path)
                dummy_val = torch.zeros(bsr_values.shape[0], device='cpu')
                dummy = torch.sparse_csr_tensor(crow_indices=crow_indices.to('cpu'),
                                                col_indices=col_indices.to('cpu'),
                                                values=dummy_val)
                dummy_coo = dummy.to_sparse(layout=torch.sparse_coo).coalesce()
                indices = dummy_coo.indices().to(input.device)
            results_shape = (n_diag_blocks, dm) if dm > 1 else (n_diag_blocks,)
            results = torch.zeros(results_shape, dtype=values.dtype, device=values.device)
            # Apply op to structurally present entries before scattering
            if op is not None:
                op(values)
                block_diags[diag_indices] = values
            results[indices[0, diag_indices]] = values
        if bsr_values.ndim > 1:
            results = torch.flatten(results, start_dim=-2, end_dim=-1)
        return results
    else:
        raise BAESparsityError(
            f"diagonal_op_ only supports square blocks with offset=0. "
            f"Got block shape ({dm}, {dn}) with offset={offset}. "
            f"For non-square blocks, convert to dense or use a different approach."
        )


def spdiags_(diagonal, offset, shape, layout):
    """
    Creates a sparse 2D tensor by placing the values from rows of diagonals along specified diagonals of the output
    tensor.

    Args:
        diagonal (Tensor): Matrix storing diagonals row-wise.
        offset (int): The diagonal in the output tensor corresponding to the main diagonal of the input tensor.
        shape (tuple[int, int]): The shape of the output tensor.
        layout (str): The layout of the output tensor.
    """
    crow_indices = torch.arange(0, shape[0] + 1, device=diagonal.device, dtype=torch.int32)
    col_indices = torch.arange(0, shape[1], device=diagonal.device, dtype=torch.int32)
    return torch.sparse_csr_tensor(crow_indices, col_indices, diagonal)


def inv_op(input):
    crow_indices = input.crow_indices() # b + 1 dimensional
    col_indices = input.col_indices() # b + 1 dimensional
    bsr_values = input.values() # 1 + 2 dimensional
    inv_values = torch.linalg.inv(bsr_values)

    return torch.sparse_bsc_tensor(crow_indices, col_indices, inv_values)

def to_cooh(input):
    crow_indices = input.crow_indices() # b + 1 dimensional
    col_indices = input.col_indices() # b + 1 dimensional
    bsr_values = input.values() # 1 + 2 dimensional
    block_shape = bsr_values.shape[-2:]
    
    dummy_csr = torch.sparse_csr_tensor(crow_indices=crow_indices.to('cpu'),
                                        col_indices=col_indices.to('cpu'),
                                        values=torch.zeros(bsr_values.shape[0], device='cpu'))
    dummy_coo = dummy_csr.to_sparse(layout=torch.sparse_coo).coalesce()
    indices = dummy_coo.indices().to(input.device)
    input_cooh = torch.sparse_coo_tensor(indices, bsr_values, [input.shape[0] // block_shape[0], input.shape[1] // block_shape[1], block_shape[0], block_shape[1]])
    return input_cooh

def to_bsr(input):
    if not input.is_coalesced():
        input = input.coalesce()
    dummy_coo = torch.sparse_coo_tensor(input.indices(), torch.zeros(input.values().shape[0], device=input.device), input.shape[:2])
    dummy_coo = dummy_coo.coalesce()
    dummy_csr = dummy_coo.to_sparse(layout=torch.sparse_csr)
    crow_indices = dummy_csr.crow_indices()
    col_indices = dummy_csr.col_indices()
    values = input.values()
    block_shape = input.shape[-2:]
    return torch.sparse_bsr_tensor(crow_indices, col_indices, values, [input.shape[0] * block_shape[0], input.shape[1] * block_shape[1]])

def add_op(input1, input2):
    input1_cooh = to_cooh(input1)
    input2_cooh = to_cooh(input2)
    input1_cooh = input1_cooh.coalesce()
    input2_cooh = input2_cooh.coalesce()
    result = input1_cooh + input2_cooh
    result = result.coalesce()
    result = to_bsr(result)
    return result

def bsr_hcat(bsr_list):
    """Horizontally concatenate BSR matrices without COO conversion.

    Concatenates multiple BSR sparse matrices along the column dimension.
    All input matrices must have the same number of block rows and
    the same block dimensions.

    Parameters
    ----------
    bsr_list : list of torch.Tensor
        List of BSR sparse tensors to concatenate horizontally.

    Returns
    -------
    torch.Tensor
        Horizontally concatenated BSR tensor.

    Raises
    ------
    BAESparsityError
        If inputs have incompatible dimensions or formats.

    Notes
    -----
    This function avoids the COO conversion overhead of torch.cat for
    sparse tensors. For J^T @ J computation with multiple jacobian blocks,
    this can save ~10ms per iteration compared to the COO path.

    Examples
    --------
    >>> J1 = jacobian(residuals, [params1])
    >>> J2 = jacobian(residuals, [params2])
    >>> J = bsr_hcat([J1, J2])  # Concatenate horizontally
    """
    if len(bsr_list) == 0:
        raise BAESparsityError("Cannot concatenate empty list of BSR tensors")
    if len(bsr_list) == 1:
        return bsr_list[0]

    first = bsr_list[0]
    if first.layout != torch.sparse_bsr:
        raise BAESparsityError(
            f"All inputs must be BSR format, got {first.layout}"
        )

    block_size = first.values().shape[-2:]
    n_block_rows = first.shape[0] // block_size[0]
    device = first.device
    dtype = first.dtype
    idx_dtype = first.crow_indices().dtype

    # Validate all inputs
    for i, bsr in enumerate(bsr_list):
        if bsr.layout != torch.sparse_bsr:
            raise BAESparsityError(
                f"Input {i} must be BSR format, got {bsr.layout}"
            )
        if bsr.values().shape[-2:] != block_size:
            raise BAESparsityError(
                f"Input {i} has block shape {bsr.values().shape[-2:]}, "
                f"expected {block_size}"
            )
        if bsr.shape[0] // block_size[0] != n_block_rows:
            raise BAESparsityError(
                f"Input {i} has {bsr.shape[0] // block_size[0]} block rows, "
                f"expected {n_block_rows}"
            )

    # Compute column block offsets for each matrix
    col_block_offsets = []
    offset = 0
    for bsr in bsr_list:
        col_block_offsets.append(offset)
        offset += bsr.shape[1] // block_size[1]

    # For each row, gather blocks from all matrices
    new_crow_list = [0]
    new_col_list = []
    new_val_list = []

    for row in range(n_block_rows):
        for i, bsr in enumerate(bsr_list):
            crow = bsr.crow_indices()
            col = bsr.col_indices()
            val = bsr.values()

            start = crow[row].item()
            end = crow[row + 1].item()

            if start < end:
                new_col_list.append(col[start:end] + col_block_offsets[i])
                new_val_list.append(val[start:end])

        # Update crow for this row
        total_in_row = sum(
            bsr.crow_indices()[row + 1].item() - bsr.crow_indices()[row].item()
            for bsr in bsr_list
        )
        new_crow_list.append(new_crow_list[-1] + total_in_row)

    # Build final tensors
    new_crow = torch.tensor(new_crow_list, dtype=idx_dtype, device=device)

    if new_col_list:
        new_col = torch.cat(new_col_list)
        new_val = torch.cat(new_val_list)
    else:
        new_col = torch.empty(0, dtype=idx_dtype, device=device)
        new_val = torch.empty(0, *block_size, dtype=dtype, device=device)

    total_cols = sum(bsr.shape[1] for bsr in bsr_list)

    return torch.sparse_bsr_tensor(
        crow_indices=new_crow,
        col_indices=new_col,
        values=new_val,
        size=(first.shape[0], total_cols)
    )


def bsr2bsc(J):
    """Convert a BSR (Block Sparse Row) tensor to BSC (Block Sparse Column).

    Parameters
    ----------
    J : torch.Tensor
        Sparse tensor in BSR format.

    Returns
    -------
    torch.Tensor
        Sparse tensor in BSC format with same data.
    """
    block_shape = J.values().shape[-2:]
    dummy_shape = [J.shape[0] // block_shape[0], J.shape[1] // block_shape[1]]
    dummy_csr = torch.sparse_csr_tensor(
        crow_indices=J.crow_indices(),
        col_indices=J.col_indices(),
        values=torch.arange(J.col_indices().shape[0], device=J.device, dtype=torch.int32),
        size=dummy_shape,
    )
    dummy_csc = dummy_csr.to_sparse_csc()
    J_bsc = torch.sparse_bsc_tensor(
        ccol_indices=dummy_csc.ccol_indices(),
        row_indices=dummy_csc.row_indices(),
        values=J.values()[dummy_csc.values()],
        size=J.shape,
    )
    return J_bsc


def bsc2bsr(J):
    """Convert a BSC (Block Sparse Column) tensor to BSR (Block Sparse Row).

    Parameters
    ----------
    J : torch.Tensor
        Sparse tensor in BSC format.

    Returns
    -------
    torch.Tensor
        Sparse tensor in BSR format with same data.
    """
    block_shape = J.values().shape[-2:]
    dummy_shape = [J.shape[0] // block_shape[0], J.shape[1] // block_shape[1]]
    dummy_csc = torch.sparse_csc_tensor(
        ccol_indices=J.ccol_indices(),
        row_indices=J.row_indices(),
        values=torch.arange(J.row_indices().shape[0], device=J.device, dtype=torch.int32),
        size=dummy_shape,
    )
    dummy_csr = dummy_csc.to_sparse_csr()
    J_bsr = torch.sparse_bsr_tensor(
        crow_indices=dummy_csr.crow_indices(),
        col_indices=dummy_csr.col_indices(),
        values=J.values()[dummy_csr.values()],
        size=J.shape,
    )
    return J_bsr


def _get_bsr_bsc_mm_kernel():
    """Get the BSR×BSC multiplication kernel, if available.

    Returns
    -------
    callable or None
        The CUDA kernel function if available, None otherwise.
    """
    try:
        from .bsr_cuda import sparse_bsr_bsc_mm as _cuda_mm
        return _cuda_mm
    except (ImportError, AttributeError):
        # Kernel not available (extension not built or missing function)
        return None


# Cache the kernel availability check
_BSR_BSC_MM_KERNEL = None
_BSR_BSC_MM_CHECKED = False


def sparse_bsr_bsc_mm(bsr, bsc):
    """Multiply BSR and BSC sparse matrices using optimized CUDA kernel.

    Computes A @ B where A is in BSR format and B is in BSC format.
    This avoids format conversion overhead compared to using CSR/cuSparse.

    Parameters
    ----------
    bsr : torch.Tensor
        Left operand in BSR (Block Sparse Row) format. Must be on CUDA.
    bsc : torch.Tensor
        Right operand in BSC (Block Sparse Column) format. Must be on CUDA.

    Returns
    -------
    torch.Tensor
        Result matrix in BSR format.

    Raises
    ------
    BAESparsityError
        If inputs are not in correct sparse format or not on CUDA.
        If the CUDA kernel is not available.

    Notes
    -----
    This function uses a custom CUDA kernel that avoids the format conversion
    overhead of the cuSparse SpGEMM path (BSR -> COO -> CSR -> SpGEMM -> CSR).
    For J^T @ J computation, pass J as BSR and convert J to BSC using bsr2bsc().

    Examples
    --------
    >>> J_bsr = ...  # Jacobian in BSR format
    >>> J_bsc = bsr2bsc(J_bsr)  # Convert to BSC for J^T
    >>> A = sparse_bsr_bsc_mm(J_bsr.mT, J_bsc.mT)  # Compute J^T @ J
    """
    global _BSR_BSC_MM_KERNEL, _BSR_BSC_MM_CHECKED

    # Check kernel availability (cached)
    if not _BSR_BSC_MM_CHECKED:
        _BSR_BSC_MM_KERNEL = _get_bsr_bsc_mm_kernel()
        _BSR_BSC_MM_CHECKED = True

    if _BSR_BSC_MM_KERNEL is None:
        raise BAESparsityError(
            "BSR×BSC CUDA kernel is not available. "
            "The bsr_cuda extension may need to be rebuilt. "
            "Falling back to CSR path in optimizer."
        )

    if bsr.layout != torch.sparse_bsr:
        raise BAESparsityError(
            f"First argument must be in BSR format, got {bsr.layout}"
        )
    if bsc.layout != torch.sparse_bsc:
        raise BAESparsityError(
            f"Second argument must be in BSC format, got {bsc.layout}"
        )
    if not bsr.is_cuda or not bsc.is_cuda:
        raise BAESparsityError(
            "Both tensors must be on CUDA device for BSR×BSC multiplication"
        )

    return _BSR_BSC_MM_KERNEL(bsr, bsc)


def is_bsr_bsc_mm_available():
    """Check if the BSR×BSC multiplication kernel is available.

    Returns
    -------
    bool
        True if the kernel is available, False otherwise.
    """
    global _BSR_BSC_MM_KERNEL, _BSR_BSC_MM_CHECKED

    if not _BSR_BSC_MM_CHECKED:
        _BSR_BSC_MM_KERNEL = _get_bsr_bsc_mm_kernel()
        _BSR_BSC_MM_CHECKED = True

    return _BSR_BSC_MM_KERNEL is not None

# Register diagonal operation for sparse CSR tensors
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sparse_lib = Library('aten', 'IMPL')
    sparse_lib.impl('diagonal', diagonal_op_, 'SparseCsrCPU')
    sparse_lib.impl('diagonal', diagonal_op_, 'SparseCsrCUDA')