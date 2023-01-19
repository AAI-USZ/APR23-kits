from torch.nn import Parameter, Module


def _tie_or_clone_weights(src, tgt, is_torch_script=False):
    if is_torch_script:
        tgt.weight = Parameter(src.weight.clone())
    else:
        tgt.weight = src.weight


def tie_weights(src: Module, *tgt: Module, is_torch_script: bool = False):
    for m in tgt:
        _tie_or_clone_weights(src=src, tgt=m, is_torch_script=is_torch_script)
