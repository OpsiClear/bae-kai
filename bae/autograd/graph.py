"""Sparse jacobian computation via operation tracing.

This module provides the core automatic differentiation functionality for
computing sparse block jacobians through operation tracing.

Key functions:
- jacobian(): Compute sparse block jacobians for traced operations
- backward(): Walk the operation trace to compute jacobians
- construct_sbt(): Construct sparse block tensors from traced data
"""

import logging

import torch
from torch import Tensor
from torch.func import jacrev

_logger = logging.getLogger(__name__)


def construct_sbt(jac_from_vmap, num, index: torch.Tensor | None, type=torch.sparse_bsc):
    """Construct a sparse block tensor from jacobian blocks.

    Parameters
    ----------
    jac_from_vmap : torch.Tensor
        Jacobian blocks computed via vmap, shape (N, block_rows, block_cols).
    num : int
        Number of columns in the block-sparse matrix.
    index : torch.Tensor, optional
        Column indices for each block. If None, uses identity indices.
    type : torch.dtype
        Sparse tensor type, either torch.sparse_bsc or torch.sparse_bsr.

    Returns
    -------
    torch.Tensor
        Sparse block tensor in BSC or BSR format.
    """
    if index is None:
        index = torch.arange(num, device=jac_from_vmap.device, dtype=torch.int32)
    n = index.shape[0] # num 2D points
    block_shape = jac_from_vmap.shape[1:]
    idx_dtype = index.dtype

    if type == torch.sparse_bsc:
        i = torch.stack([torch.arange(n, dtype=index.dtype, device=index.device), index])
        dummy_val = torch.arange(n, device=index.device, dtype=torch.int32)
        dummy_coo = torch.sparse_coo_tensor(i, dummy_val, size=(n, num), device=index.device, dtype=torch.int32)
        dummy_csc = dummy_coo.coalesce().to_sparse_csc()
        return torch.sparse_bsc_tensor(ccol_indices=dummy_csc.ccol_indices().to(torch.int32), 
                                    row_indices=dummy_csc.row_indices().to(torch.int32),
                                    values = jac_from_vmap[dummy_csc.values()],
                                    size = (n * block_shape[0], num * block_shape[1]),
                                    device=index.device, dtype=jac_from_vmap.dtype)
    elif type == torch.sparse_bsr:
        return torch.sparse_bsr_tensor(col_indices=index, 
                                    crow_indices=torch.arange(n + 1, device=index.device, dtype=idx_dtype),
                                    values = jac_from_vmap,
                                    size = (n * block_shape[0], num * block_shape[1]),
                                    device=index.device, dtype=jac_from_vmap.dtype)

def amend_trace(arg, jac_trace: tuple):
    if hasattr(arg, 'jactrace'):  # convert to sparse_bsr needed for accumulation
        if type(arg.jactrace) is tuple and type(jac_trace) is tuple:
            if arg.jactrace[0] is None and jac_trace[0] is None:
                arg.jactrace = (None, arg.jactrace[1] + jac_trace[1])
                return 
        if type(arg.jactrace) is tuple:
            arg.jactrace = construct_sbt(arg.jactrace[1], arg.shape[0], arg.jactrace[0], type=torch.sparse_bsr)
        if type(jac_trace) is tuple:
            jac_trace = construct_sbt(jac_trace[1], arg.shape[0], jac_trace[0], type=torch.sparse_bsr)
        arg.jactrace = arg.jactrace + jac_trace
    else:
        arg.jactrace = jac_trace

def update_from_trace(bsrt: torch.Tensor, arg, new_col: torch.Tensor | None = None, new_val: torch.Tensor | None = None):
    if new_col is not None:
        jac_trace = torch.sparse_bsr_tensor(
                col_indices=new_col, 
                crow_indices=bsrt.crow_indices(),
                values=bsrt.values(),
                size=(bsrt.shape[0], arg.shape[0] * bsrt.values().shape[-1]),
                device=bsrt.device,
            )
    if new_val is not None:
        jac_trace = torch.sparse_bsr_tensor(
                col_indices=bsrt.col_indices(), 
                crow_indices=bsrt.crow_indices(),
                values=new_val,
                size=(bsrt.shape[0], arg.shape[0] * new_val.shape[-1]),
                device=bsrt.device,
            )
    return jac_trace

def backward(output_):
    """Walk the operation trace backward to compute jacobians.

    Recursively processes the operation trace attached to output_, computing
    jacobian blocks for each traced operation. Handles both 'map' operations
    (function applications) and 'index' operations (tensor indexing).

    Parameters
    ----------
    output_ : TrackingTensor
        Output tensor with attached operation trace (optrace attribute).

    Notes
    -----
    This function modifies parameters in-place, attaching jactrace attributes
    containing jacobian information. The jacobian() function collects these
    traces and constructs the final sparse block jacobian matrices.
    """
    if output_.optrace[id(output_)][0] == 'map':
        func = output_.optrace[id(output_)][1]
        args = output_.optrace[id(output_)][2]
        argnums = tuple(idx for idx, arg in enumerate(args) if hasattr(arg, 'optrace') or isinstance(arg, torch.nn.Parameter))
        if len(argnums) == 0:
            _logger.warning("No upstream parameters to compute jacobian")
            return
        jac_blocks = torch.vmap(jacrev(func, argnums=argnums))(*args)
        for jacidx, argidx in enumerate(argnums):
            jac_block = jac_blocks[jacidx]
            arg = args[argidx]
            assert jac_block.ndim == 3, "`func` is not properly vectorized in `torch.vmap`"
            # Jacobian blocks are kept as 3D tensors (batch, rows, cols) to preserve
            # block structure for BSR format. Flattening would lose sparsity benefits.
            if not hasattr(output_, 'jactrace'):  # check for upstream jacobian
                jac_trace = (None, jac_block)  # leave None for identity indices
            else:
                indices = None
                if type(output_.jactrace) is tuple:
                    indices = output_.jactrace[0]
                    jac_ustrm = output_.jactrace[1]
                elif type(output_.jactrace) is torch.Tensor and output_[jacidx].jactrace.layout == torch.sparse_bsr:
                    indices = output_.jactrace.col_indices()
                    jac_ustrm = output_.jactrace.values()

                if indices is not None:
                    jac_block = jac_block[indices]
                jac_block = jac_ustrm @ jac_block
                
                if type(output_.jactrace) is tuple:
                    jac_trace = (indices, jac_block)
                elif type(output_.jactrace) is torch.Tensor and output_.jactrace.layout == torch.sparse_bsr:
                    jac_trace = update_from_trace(output_.jactrace, arg, new_val=jac_block)
            amend_trace(arg, jac_trace)
        for argidx in argnums:
            if isinstance(args[argidx], torch.Tensor) and hasattr(args[argidx], 'optrace'):
                backward(args[argidx])


    elif output_.optrace[id(output_)][0] == 'index':
        index = output_.optrace[id(output_)][1]
        arg = output_.optrace[id(output_)][2]

        # If the last operation is indexing, there is no downstream map op to
        # populate Jacobian values. In this case, the Jacobian block values are
        # identity matrices placed at the indexed columns.
        if not hasattr(output_, 'jactrace'):
            if output_.ndim == 1:
                eye_blocks = torch.ones((output_.shape[0], 1, 1), device=output_.device, dtype=output_.dtype)
            else:
                block_dim = output_.shape[-1]
                eye = torch.eye(block_dim, device=output_.device, dtype=output_.dtype)
                eye_blocks = eye.unsqueeze(0).repeat(output_.shape[0], 1, 1)
            output_.jactrace = (None, eye_blocks)

        if type(output_.jactrace) is tuple:
            if output_.jactrace[0] is not None:
                upstream_index = output_.jactrace[0]
                index = index[upstream_index]
                jac_trace = (index, output_.jactrace[1])
            elif output_.jactrace[0] is None:
                jac_trace = (index, output_.jactrace[1])
        elif type(output_.jactrace) is torch.Tensor and output_.jactrace.layout == torch.sparse_bsr:
            upstream_index = output_.jactrace.col_indices()
            index = index[upstream_index]
            jac_trace = update_from_trace(output_.jactrace, arg, new_col=index)
            
        amend_trace(arg, jac_trace)
        if isinstance(arg, torch.Tensor) and hasattr(arg, 'optrace'):
            backward(arg)


def jacobian(output: Tensor, params: list[torch.nn.Parameter]) -> list[Tensor]:
    """Compute sparse block jacobians for traced operations.

    Walks the operation trace attached to output and computes sparse block
    jacobians with respect to each parameter in params.

    Parameters
    ----------
    output : TrackingTensor
        Output tensor from traced operations. Must have optrace attribute
        with last operation being 'map' or 'index'.
    params : list of torch.nn.Parameter
        List of parameters to compute jacobians for.

    Returns
    -------
    list of torch.Tensor
        List of sparse BSR jacobian matrices, one per parameter.
        Each jacobian has shape (output_size, param_size) with block
        structure determined by the traced operations.

    Raises
    ------
    AssertionError
        If the last operation in the trace is not 'map' or 'index'.

    Notes
    -----
    For SE(3) parameters (with trim_SE3_grad=True), the jacobian is computed
    in the 6-DOF tangent space, removing the redundant 7th element.

    Examples
    --------
    >>> from bae.autograd import jacobian, TrackingTensor
    >>> x = TrackingTensor(data, requires_grad=True)
    >>> y = traced_function(x)
    >>> J = jacobian(y, [param1, param2])
    """
    assert output.optrace[id(output)][0] in ('map', 'index'), "Unsupported last operation in compute graph"
    backward(output)
    res = []
    for param in params:
        if hasattr(param, 'jactrace'):
            if getattr(param, 'trim_SE3_grad', False):
                if isinstance(param.jactrace, tuple):
                    values = param.jactrace[1]
                elif isinstance(param.jactrace, torch.Tensor) and param.jactrace.layout == torch.sparse_bsr:
                    values = param.jactrace.values()
                else:
                    values = param.jactrace

                if values.shape[-1] == 7:
                    values = values[..., :6]
                else:
                    values = torch.cat([values[..., :6], values[..., 7:]], dim=-1)
                
                if isinstance(param.jactrace, tuple):
                    param.jactrace = (param.jactrace[0], values)
                elif isinstance(param.jactrace, torch.Tensor) and param.jactrace.layout == torch.sparse_bsr:
                    param.jactrace = torch.sparse_bsr_tensor(
                        col_indices=param.jactrace.col_indices(), 
                        crow_indices=param.jactrace.crow_indices(),
                        values=values,
                        size=(param.jactrace.shape[0], param.shape[0] * values.shape[-1]),
                        device=param.device,
                    )
                else:
                    param.jactrace = values
            if type(param.jactrace) is tuple:
                param.jactrace = construct_sbt(param.jactrace[1], param.shape[0], param.jactrace[0], type=torch.sparse_bsr)
            res.append(param.jactrace)
            delattr(param, 'jactrace')
            
    return res
