from tiktoken.load import load_tiktoken_bpe

ENDOFTEXT = "<|endoftext|>"
IM_PREFIX = "<|im_start|>"
IM_MIDDLE = "<|im_sep|>"
IM_SUFFIX = "<|im_end|>"
IM_NEWLINES = "\n\n"

def p50k_im():
    mergeable_ranks = load_tiktoken_bpe(
        "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken"
    )
    special_tokens = {ENDOFTEXT: 50256, IM_PREFIX: 50281, IM_MIDDLE: 50282, IM_SUFFIX: 50283}
    mergeable_ranks[IM_NEWLINES.encode()] = 50284
    return {
        "name": "p50k_im",
        "pat_str": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
    }

ENCODING_CONSTRUCTORS = {
    "p50k_im": p50k_im,
}