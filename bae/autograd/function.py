import torch

WHITELISTED_MAPS = (torch._C.TensorBase.__add__,
                    torch._C.TensorBase.__sub__,
                    torch._C.TensorBase.__mul__,
                    torch._C.TensorBase.__div__,
                    torch._C.TensorBase.add,
                    torch._C.TensorBase.sub,
                    torch._C.TensorBase.mul,)

# =============================================================================
# Class: TrackingTensor
# A custom subclass of torch.Tensor that tracks the indices used for slicing.
# When an instance is sliced via __getitem__, it records the provided index.
# =============================================================================
class TrackingTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        if isinstance(data, torch.Tensor):
            instance = torch.Tensor._make_subclass(cls, data, *args, **kwargs)
        else:
            instance = torch.Tensor._make_subclass(cls, torch.as_tensor(data), *args, **kwargs)
        return instance


    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        result = super(TrackingTensor, cls).__torch_function__(func, types, args=args, kwargs=kwargs)

        if isinstance(result, torch.Tensor) and getattr(args[0], '_active', True):
            if (func == torch._C.TensorBase.__getitem__) and isinstance(args[1], torch.Tensor):
                if not hasattr(result, 'optrace'):
                    result.optrace = {}
                index_edge = ("index", args[1], args[0])
                result.optrace[id(result)] = index_edge
            elif func in WHITELISTED_MAPS:
                merged_optrace = {}
                for arg in args:
                    if isinstance(arg, torch.Tensor) and hasattr(arg, 'optrace'):
                        merged_optrace.update(arg.optrace)

                merged_optrace[id(result)] = ("map", func, args)
                result.optrace = merged_optrace
        return result

    def __getitem__(self, index):
        result = super().__getitem__(index)
        return result

    def tensor(self) -> torch.Tensor:
        return torch.Tensor.as_subclass(self, torch.Tensor)


# =============================================================================
# Function: index_transform
# Wraps a tensor indexing operation to attach operation trace metadata.
# =============================================================================
def index_transform(tensor, index):
    result = tensor[index]
    if not hasattr(result, 'optrace'):
        result.optrace = {}
    index_edge = ("index", index, tensor)
    result.optrace[id(result)] = index_edge
    return result


# =============================================================================
# Decorator: map_transform
# Wraps a function to apply a map transformation and merge operation traces.
# =============================================================================
def map_transform(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # map edge (edge_type, func, [input_args])
        merged_optrace = {}
        for arg in args:
            if isinstance(arg, torch.Tensor) and hasattr(arg, 'optrace'):
                merged_optrace.update(arg.optrace)

        merged_optrace[id(result)] = ("map", func, args)
        result.optrace = merged_optrace
        return result
    return wrapper
