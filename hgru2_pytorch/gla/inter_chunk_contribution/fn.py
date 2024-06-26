from .chunk_scan_triton_full import Chunk_memory_update_full
from .chunk_scan_triton_no_decay import Chunk_memory_update_no_decay
from .chunk_scan_triton_only_gk import Chunk_memory_update_only_gk
from .chunk_scan_triton_only_gv import Chunk_memory_update_only_gv
from .preprocess_cumsum_gk import PreprocessCumSum_GK
from .preprocess_cumsum_gv import PreprocessCumSum_GV


def inter_chunk_onc(query, key, value, gk, gv, normalizer_gk=16, clam_min=-3):

    if gk is not None:
        g_key_cumsum, reduce_key, q_exp, g_key_last_exp = PreprocessCumSum_GK.apply(
            query, key, gk, normalizer_gk, clam_min
        )
    else:
        reduce_key = key
        q_exp = None
        g_key_cumsum = None
        g_key_last_exp = None

    # gv_cumsum, v_reduce, gv_cumsum_exp, gv_last_exp

    if gv is not None:
        (
            g_value_cumsum,
            reduce_value,
            g_value_cumsum_exp,
            g_value_last_exp,
        ) = PreprocessCumSum_GV.apply(value, gv, normalizer_gv, clam_min)
    else:
        reduce_value = value
        g_value_cumsum = None
        g_value_last_exp = None

    to_add = reduce_key.transpose(-1, -2) @ reduce_value

    if gk is not None and gv is not None:
        memory_cache = Chunk_memory_update_full.apply(
            g_key_last_exp, g_value_last_exp, to_add
        )
        inter_chunk_contribution = ((q_exp) @ memory_cache) * g_value_cumsum_exp
    elif gk is None and gv is not None:
        memory_cache = Chunk_memory_update_only_gv.apply(g_value_last_exp, to_add)
        inter_chunk_contribution = ((query) @ memory_cache) * g_value_cumsum_exp
    elif gk is not None and gv is None:
        memory_cache = Chunk_memory_update_only_gk.apply(g_key_last_exp, to_add)
        inter_chunk_contribution = (q_exp) @ memory_cache
    else:
        memory_cache = Chunk_memory_update_no_decay.apply(to_add)
        inter_chunk_contribution = (query) @ memory_cache

    return g_key_cumsum, g_value_cumsum, inter_chunk_contribution
