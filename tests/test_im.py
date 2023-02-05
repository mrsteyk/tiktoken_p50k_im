import tiktoken

def test_whitespaces():
    enc = tiktoken.get_encoding("p50k_im")
    for i in range(1, 26):
        assert len(enc.encode(' '*i)) == 1, f"{i} whitespaces isn't one token"

def test_im_tokens():
    enc = tiktoken.get_encoding("p50k_im")
    
    start = enc.encode("<|im_start|>", allowed_special="all")
    assert len(start) == 1
    start = start[0]

    end = enc.encode("<|im_end|>", allowed_special="all")
    assert len(end) == 1
    end = end[0]

    sep = enc.encode("<|im_sep|>", allowed_special="all")
    assert len(sep) == 1
    sep = sep[0]

    assert sep > end, "IM's end must come before sep!"