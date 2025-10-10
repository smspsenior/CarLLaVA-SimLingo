import os
import torch
from typing import Any, Dict
from transformers import PreTrainedModel

def _local_rank() -> int:
    for k in ("LOCAL_RANK", "DEEPSPEED_LOCAL_RANK"):
        v = os.environ.get(k)
        if v and v.isdigit():
            return int(v)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        try:
            r = int(os.environ["RANK"])
            w = max(1, int(os.environ["WORLD_SIZE"]))
            return r % w
        except Exception:
            pass
    return 0

def streaming_from_pretrained(
    model_cls, 
    variant: str,
    dtype: torch.dtype = torch.float16,
    extra_kwargs: Dict[str, Any] = None,
) -> PreTrainedModel:
    extra_kwargs = dict(extra_kwargs or {})
    lr = _local_rank()
    kwargs = dict(
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device_map={"": f"cuda:{lr}"},
        trust_remote_code=True,
    )
    kwargs.update(extra_kwargs)
    return model_cls.from_pretrained(variant, **kwargs).eval()

try:
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from transformers import AutoConfig
    def dispatch_from_shards_accelerate(
        model_cls,
        variant: str,
        dtype: torch.dtype = torch.float16,
        extra_config_kwargs: Dict[str, Any] = None,
    ) -> PreTrainedModel:
        lr = _local_rank()
        cfg = AutoConfig.from_pretrained(variant, trust_remote_code=True, **(extra_config_kwargs or {}))
        with init_empty_weights():
            model = model_cls.from_config(cfg, trust_remote_code=True)
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=variant,
            device_map={"": f"cuda:{lr}"},
            dtype=dtype,
            offload_state_dict=True, 
        )
        return model.eval()
except Exception:
    pass

