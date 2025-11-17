from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    # @formatter:off
    # Model params for ./stories15M.model.npz
    dim: int = 288  # D
    n_layers: int = 6
    n_heads: int = 6  # QHN, HN, HD = 48
    n_kv_heads: Optional[int] = None  # KVHN = 6
    vocab_size: int = 32000  # VS
    max_seq_len: int = 256  # M
    max_new_tokens: int = 150
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    max_batch_size: int = 1
    dtype: str = "float32"
