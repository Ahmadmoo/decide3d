import struct
import numpy as np

# minimal TFRecord + tf.train.Example reader, no tensorflow needed
# Example > Features > map<key, Feature> > bytes_list | float_list


def varint(b, i):
    r, s = 0, 0
    while True:
        c = b[i]; i += 1
        r |= (c & 127) << s; s += 7
        if c < 128: return r, i


def fields(b):
    i, n = 0, len(b)
    while i < n:
        tag, i = varint(b, i)
        no, wire = tag >> 3, tag & 7
        if wire == 2:
            size, i = varint(b, i)
            yield no, b[i:i + size]; i += size
        elif wire == 0: _, i = varint(b, i)
        elif wire == 1: i += 8
        elif wire == 5: i += 4
        else: raise ValueError(f"wire type {wire}")


def parse_example(rec):
    out = {}
    for _, feats in fields(memoryview(rec)):
        for _, entry in fields(feats):
            key, val = None, None
            for no, x in fields(entry):
                if no == 1: key = bytes(x).decode()
                else: val = x
            for kind, lst in fields(val):
                if kind == 1: out[key] = [x for _, x in fields(lst)]
                elif kind == 2: out[key] = np.concatenate([np.frombuffer(x, "<f4") for _, x in fields(lst)]) if len(lst) else np.zeros(0, "<f4")
    return out


def records(path):
    with open(path, "rb") as f:
        while True:
            head = f.read(12)
            if not head: return
            if len(head) < 12: raise IOError(f"truncated {path}")
            n = struct.unpack("<Q", head[:8])[0]
            data = f.read(n)
            if len(data) < n or len(f.read(4)) < 4: raise IOError(f"truncated {path}")
            yield data
