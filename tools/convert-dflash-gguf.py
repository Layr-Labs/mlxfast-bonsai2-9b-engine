#!/usr/bin/env python3
# Deterministic Q8_0 GGUF -> MLX safetensors converter for the Bonsai 2 9B
# DFlash 2 block drafter.
#
# SCOPE AND PROVENANCE. David sanctioned "our own conversion" of ONE drafter
# (memory dflash2-drafter-not-platform-specific.md, 2026-10-14). The pinned
# SOURCE stays Prism's GGUF bytes
# (prism-ml/Ternary-Bonsai-2-9B-gguf: Ternary-Bonsai-2-9B-DFlash2-Q8_0.gguf,
# sha256 95149abd..., 1574233856 B, commit 608e89c3); this script is deterministic engine code
# whose output sha256 is recorded and asserted. The waiver does NOT extend to
# the trunk, the MTP head or any other weight.
#
# WHAT IT WRITES. One model.safetensors whose keys are the MLX module parameter
# paths of Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/DFlash2Draft.swift:
#
#   * Every Q8_0 LINEAR weight becomes an MLX affine 8-bit, group-32 quantized
#     linear, re-encoded without loss. A Q8_0 block is 32 int8 values and one
#     FP16 scale, value = scale * int8. MLX affine stores q in [0, 255] and
#     computes value = scale * q + bias. With q = int8 + 128, the scale = the
#     block's FP16 scale and bias = -128 * scale, the MLX value is
#     scale * (int8 + 128) - 128 * scale = scale * int8 exactly. The scales and
#     biases are written as FP32: FP32 holds every FP16 scale and every
#     -128 * scale exactly, and the drafter's quantized matmuls then run in FP32.
#     The MLX group axis is the input axis, which is the Q8_0 block axis, so
#     each MLX group is one Q8_0 block. This is the mapping MLX's own GGUF
#     loader uses (Vendor/mlx-swift/Source/Cmlx/mlx/mlx/io/gguf_quants.cpp,
#     extract_q8_0_data: q = byte ^ 0x80, bias = -128 * scale), with FP32 in
#     place of its FP16 scales and biases.
#   * The two selector codebooks (Q8_0, but raw parameters, not linears) are
#     dequantised (scale * int8 in FP32) and written as BF16.
#   * The norms and the convolution base kernels (F32 in the GGUF) are written
#     as BF16. The drafter's parameter dtype follows norm.weight, so it is BF16;
#     FP16 overflows inside the drafter trunk (NaN logits, lane BH 2026-10-14).
#
# LAYOUT. The MLX shape is the GGUF logical (reversed-ne) shape, so no
# transpose or permute is applied: a GGUF linear weight stored ne=[in,out] is
# [out,in] = MLX Linear.weight, the conv base ne=[hidden,kernel,tap] is
# [tap,kernel,hidden] = module base_kernel, and the selector codebooks
# ne=[rank,vocab] are [vocab,rank] = module codebook.
#
# DETERMINISM. Python standard library only (no numpy: the box and participant
# setups need nothing beyond python3). No threads, no RNG, no file-order
# dependence (the safetensors header keys are sorted). Given the pinned GGUF
# the output is byte-identical on every run.
#
# Run with -I so a planted module on the path cannot be imported:
#   python3 -I tools/convert-dflash-gguf.py --gguf <PATH> --out-dir <DIR>

import argparse
import array
import hashlib
import json
import math
import os
import struct
import sys

# ggml tensor type ids used by this drafter.
GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8

# GGUF metadata value type ids.
(GT_U8, GT_I8, GT_U16, GT_I16, GT_U32, GT_I32, GT_F32, GT_BOOL, GT_STR, GT_ARR,
 GT_U64, GT_I64, GT_F64) = range(13)

_SCALAR_FMT = {
    GT_U8: "B", GT_I8: "b", GT_BOOL: "B",
    GT_U16: "H", GT_I16: "h",
    GT_U32: "I", GT_I32: "i", GT_F32: "f",
    GT_U64: "Q", GT_I64: "q", GT_F64: "d",
}

_SCALAR_SIZE = {
    GT_U8: 1, GT_I8: 1, GT_BOOL: 1,
    GT_U16: 2, GT_I16: 2,
    GT_U32: 4, GT_I32: 4, GT_F32: 4,
    GT_U64: 8, GT_I64: 8, GT_F64: 8,
}


class Cursor:
    """A little-endian reader over an in-memory GGUF header region."""

    def __init__(self, buf, offset=0):
        self.buf = buf
        self.off = offset

    def u32(self):
        v = struct.unpack_from("<I", self.buf, self.off)[0]
        self.off += 4
        return v

    def u64(self):
        v = struct.unpack_from("<Q", self.buf, self.off)[0]
        self.off += 8
        return v

    def gstr(self):
        n = self.u64()
        s = self.buf[self.off:self.off + n]
        self.off += n
        return bytes(s)

    def skip_value(self, vtype):
        if vtype == GT_STR:
            n = self.u64()
            self.off += n
        elif vtype == GT_ARR:
            elem_type = self.u32()
            count = self.u64()
            if elem_type == GT_STR:
                for _ in range(count):
                    n = self.u64()
                    self.off += n
            elif elem_type == GT_ARR:
                raise ValueError("nested arrays are not supported")
            else:
                self.off += _SCALAR_SIZE[elem_type] * count
        else:
            self.off += _SCALAR_SIZE[vtype]

    def read_value(self, vtype):
        """Read one metadata value. An array of strings longer than 64 entries
        (the tokenizer vocabulary) is skipped and returned as its length."""
        if vtype in _SCALAR_FMT:
            fmt = _SCALAR_FMT[vtype]
            v = struct.unpack_from("<" + fmt, self.buf, self.off)[0]
            self.off += struct.calcsize(fmt)
            return bool(v) if vtype == GT_BOOL else v
        if vtype == GT_STR:
            return self.gstr().decode("utf-8")
        if vtype == GT_ARR:
            elem_type = self.u32()
            count = self.u64()
            if elem_type == GT_STR and count > 64:
                for _ in range(count):
                    n = self.u64()
                    self.off += n
                return count
            if elem_type == GT_ARR:
                raise ValueError("nested arrays are not supported")
            return [self.read_value(elem_type) for _ in range(count)]
        raise ValueError("unknown GGUF value type %d" % vtype)


def align_up(x, a):
    return (x + a - 1) // a * a


def parse_gguf(buf):
    """Return (alignment, [(name, ne, ggml_type, offset)], data_start, metadata)."""
    cur = Cursor(buf)
    magic = cur.u32()
    if magic != 0x46554747:  # "GGUF"
        raise ValueError("not a GGUF file (bad magic)")
    version = cur.u32()
    if version != 3:
        raise ValueError("unsupported GGUF version %d" % version)
    tensor_count = cur.u64()
    kv_count = cur.u64()

    metadata = {}
    for _ in range(kv_count):
        key = cur.gstr().decode("utf-8")
        vtype = cur.u32()
        if key == "tokenizer.chat_template" or key.startswith("tokenizer.ggml.merges"):
            cur.skip_value(vtype)
        else:
            metadata[key] = cur.read_value(vtype)
    alignment = metadata.get("general.alignment", 32)

    tensors = []
    for _ in range(tensor_count):
        name = cur.gstr().decode("utf-8")
        n_dims = cur.u32()
        ne = [cur.u64() for _ in range(n_dims)]
        ggml_type = cur.u32()
        offset = cur.u64()
        tensors.append((name, ne, ggml_type, offset))

    data_start = align_up(cur.off, alignment)
    return alignment, tensors, data_start, metadata


def tensor_numel(ne):
    numel = 1
    for d in ne:
        numel *= int(d)
    return numel


def logical_shape(ne):
    # GGUF stores ne[0] fastest; the logical (PyTorch/MLX) shape is ne reversed,
    # and a C-order layout of ne[::-1] places ne[0] on the last (fastest) axis.
    # The raw GGUF bytes are therefore already in MLX row-major order.
    return [int(d) for d in ne[::-1]]


# Byte tables. XOR 0x80 maps an int8 byte to int8 + 128 as uint8.
_XOR80 = bytes((b ^ 0x80) for b in range(256))
_INT8 = [b - 256 if b >= 128 else b for b in range(256)]


def bf16_bits(x):
    """Round-to-nearest-even fp32 -> bf16 bit pattern of one Python float."""
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


def bf16_bytes(values):
    """BF16 little-endian bytes of a sequence of floats (each fp32-exact)."""
    f32 = array.array("f", values)
    if sys.byteorder != "little":
        f32.byteswap()
    u32 = array.array("I")
    u32.frombytes(f32.tobytes())
    out = array.array("H", (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF for u in u32))
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


def f32_bytes(values):
    out = array.array("f", values)
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


def q8_0_split(buf, data_start, ne, offset):
    """Split a Q8_0 tensor into (fp16 scales as floats, raw int8 bytes)."""
    numel = tensor_numel(ne)
    if numel % 32 != 0:
        raise ValueError("Q8_0 tensor numel %d not divisible by 32" % numel)
    n = numel // 32
    base = data_start + offset
    region = buf[base:base + 34 * n]
    scale_bytes = bytearray(2 * n)
    scale_bytes[0::2] = region[0::34]
    scale_bytes[1::2] = region[1::34]
    scales = struct.unpack("<%de" % n, scale_bytes)
    if not all(math.isfinite(x) for x in scales):
        raise ValueError("Q8_0 tensor has a non-finite scale")
    qs = bytearray(32 * n)
    for j in range(32):
        qs[j::32] = region[2 + j::34]
    return scales, bytes(qs)


def affine8_from_q8_0(buf, data_start, ne, offset):
    """Re-encode a Q8_0 linear weight as MLX affine 8-bit group 32, losslessly.

    Returns {suffix: (safetensors dtype, shape, bytes)} for weight/scales/biases.
    """
    out_dim, in_dim = logical_shape(ne)
    if in_dim % 32 != 0:
        raise ValueError("linear input dim %d not divisible by 32" % in_dim)
    scales, qs = q8_0_split(buf, data_start, ne, offset)
    # 8-bit packing: four values per uint32, value i of a word in bits
    # [8*i, 8*i+8). That is the little-endian byte order of q itself.
    q = qs.translate(_XOR80)
    groups = in_dim // 32
    return {
        "weight": ("U32", [out_dim, in_dim // 4], q),
        "scales": ("F32", [out_dim, groups], f32_bytes(scales)),
        "biases": ("F32", [out_dim, groups], f32_bytes(-128.0 * x for x in scales)),
    }


def bf16_from_q8_0(buf, data_start, ne, offset):
    """Dequantise a Q8_0 tensor (scale * int8, exact in fp32) to BF16 bytes."""
    scales, qs = q8_0_split(buf, data_start, ne, offset)
    lo_tables = {}
    hi_tables = {}
    lo = []
    hi = []
    for i, scale in enumerate(scales):
        if scale not in lo_tables:
            bits = [bf16_bits(scale * v) for v in _INT8]
            lo_tables[scale] = bytes(b & 0xFF for b in bits)
            hi_tables[scale] = bytes(b >> 8 for b in bits)
        block = qs[32 * i:32 * i + 32]
        lo.append(block.translate(lo_tables[scale]))
        hi.append(block.translate(hi_tables[scale]))
    out = bytearray(2 * len(qs))
    out[0::2] = b"".join(lo)
    out[1::2] = b"".join(hi)
    return bytes(out)


def bf16_from_float(buf, data_start, ne, ggml_type, offset):
    numel = tensor_numel(ne)
    base = data_start + offset
    if ggml_type == GGML_F32:
        values = struct.unpack_from("<%df" % numel, buf, base)
    elif ggml_type == GGML_F16:
        values = struct.unpack_from("<%de" % numel, buf, base)
    else:
        raise ValueError("tensor ggml_type %d is not supported" % ggml_type)
    return bf16_bytes(values)


def mlx_key(gguf_name):
    """Map one GGUF tensor name to its MLX DFlash2DraftModel parameter path."""
    direct = {
        "fc.weight": "fc.weight",
        "enc.output_norm.weight": "hidden_norm.weight",
        "output_norm.weight": "norm.weight",
        "selector_hidden.weight": "candidate_selector.hidden_projection.weight",
        "selector_predecessor.weight": "candidate_selector.predecessor_codebook",
        "selector_successor.weight": "candidate_selector.successor_codebook",
    }
    if gguf_name in direct:
        return direct[gguf_name]
    if not gguf_name.startswith("blk."):
        raise ValueError("unmapped GGUF tensor %r" % gguf_name)
    _, idx, rest = gguf_name.split(".", 2)
    per_layer = {
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
        "attn_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "attn_conv_base": "attention_conv.base_kernel",
        "attn_conv_proj.weight": "attention_conv.kernel_projection.weight",
        "ffn_conv_base": "mlp_conv.base_kernel",
        "ffn_conv_proj.weight": "mlp_conv.kernel_projection.weight",
    }
    if rest not in per_layer:
        raise ValueError("unmapped per-block GGUF tensor %r" % gguf_name)
    return "layers.%s.%s" % (idx, per_layer[rest])


def write_safetensors(path, tensors):
    """Write a minimal, valid safetensors file.

    tensors: dict name -> (safetensors dtype, shape, raw little-endian bytes).
    """
    header = {}
    offset = 0
    for name in sorted(tensors.keys()):
        st_dtype, shape, data = tensors[name]
        header[name] = {
            "dtype": st_dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    header_json = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for name in sorted(tensors.keys()):
            f.write(tensors[name][2])


def config_mismatches(meta, config):
    """Compare the drafter config.json with the GGUF header. Every geometry
    value of the config is read off the header here; a difference is named."""
    a = "dflash."
    rope = config.get("rope_parameters", {})
    block = config.get("dflash_config", {})
    pattern = meta.get(a + "attention.sliding_window_pattern")
    want = [
        ("hidden_size", config.get("hidden_size"), meta.get(a + "embedding_length")),
        ("intermediate_size", config.get("intermediate_size"), meta.get(a + "feed_forward_length")),
        ("num_hidden_layers", config.get("num_hidden_layers"), meta.get(a + "block_count")),
        ("num_attention_heads", config.get("num_attention_heads"), meta.get(a + "attention.head_count")),
        ("num_key_value_heads", config.get("num_key_value_heads"), meta.get(a + "attention.head_count_kv")),
        ("head_dim", config.get("head_dim"), meta.get(a + "attention.key_length")),
        ("sliding_window", config.get("sliding_window"), meta.get(a + "attention.sliding_window")),
        ("layer_types", config.get("layer_types"),
         None if pattern is None else ["sliding_attention" if x else "full_attention" for x in pattern]),
        ("vocab_size", config.get("vocab_size"), meta.get("tokenizer.ggml.tokens")),
        ("rope_parameters.rope_theta", rope.get("rope_theta"), meta.get(a + "rope.freq_base")),
        ("rope_parameters.rope_type", rope.get("rope_type", "default"), meta.get(a + "rope.scaling.type", "default")),
        ("rope_parameters.factor", rope.get("factor"), meta.get(a + "rope.scaling.factor")),
        ("rope_parameters.original_max_position_embeddings", rope.get("original_max_position_embeddings"),
         meta.get(a + "rope.scaling.original_context_length")),
        ("dflash_config.block_size", block.get("block_size"), meta.get(a + "block_size")),
        ("dflash_config.mask_token_id", block.get("mask_token_id"), meta.get("tokenizer.ggml.mask_token_id")),
        ("dflash_config.target_layer_ids", block.get("target_layer_ids"), meta.get(a + "target_layers")),
        ("dflash_config.conv_kernel_size", block.get("conv_kernel_size"), meta.get(a + "conv_kernel_size")),
        ("dflash_config.conv_group_size", block.get("conv_group_size"), meta.get(a + "conv_group_size")),
        ("dflash_config.selector_rank", block.get("selector_rank"), meta.get(a + "selector_rank")),
        ("dflash_config.selector_top_k", block.get("selector_top_k"), meta.get(a + "selector_top_k")),
    ]
    out = []
    for name, have, header in want:
        same = (have == header) if not isinstance(header, float) else (
            isinstance(have, (int, float)) and abs(have - header) <= 1e-6 * max(1.0, abs(header)))
        if not same:
            out.append("%s: config %r, GGUF header %r" % (name, have, header))
    eps = meta.get(a + "attention.layer_norm_rms_epsilon")
    if eps is None or abs(config.get("rms_norm_eps", 0) - eps) > 1e-9:
        out.append("rms_norm_eps: config %r, GGUF header %r" % (config.get("rms_norm_eps"), eps))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expect-source-sha256", default=None,
                    help="assert the GGUF file sha256 before converting")
    ap.add_argument("--config", default=None,
                    help="refuse unless this config.json matches the GGUF header")
    ap.add_argument("--dump", action="store_true",
                    help="print the tensor table (name, ne, ggml_type) and exit")
    args = ap.parse_args()

    with open(args.gguf, "rb") as f:
        buf = f.read()

    src_sha = hashlib.sha256(buf).hexdigest()
    if args.expect_source_sha256 and src_sha != args.expect_source_sha256:
        print("source sha256 mismatch: got %s expected %s" %
              (src_sha, args.expect_source_sha256), file=sys.stderr)
        return 2

    alignment, tensors, data_start, metadata = parse_gguf(buf)

    if args.config:
        with open(args.config, "rb") as f:
            config = json.load(f)
        bad = config_mismatches(metadata, config)
        if bad:
            for line in bad:
                print("config mismatch: %s" % line, file=sys.stderr)
            return 3

    if args.dump:
        print("alignment=%d data_start=%d tensor_count=%d source_sha256=%s" %
              (alignment, data_start, len(tensors), src_sha))
        for key in sorted(k for k in metadata if k.startswith("dflash.")):
            print("%s = %r" % (key, metadata[key]))
        for name, ne, gt, off in tensors:
            print("%-40s ne=%s ggml_type=%d offset=%d" % (name, ne, gt, off))
        return 0

    out = {}

    def put(key, value):
        if key in out:
            raise ValueError("duplicate MLX key %r" % key)
        out[key] = value

    for name, ne, gt, off in tensors:
        key = mlx_key(name)
        if gt == GGML_Q8_0 and key.endswith(".weight"):
            # A linear: lossless MLX affine 8-bit group 32.
            module = key[: -len(".weight")]
            for suffix, value in affine8_from_q8_0(buf, data_start, ne, off).items():
                put("%s.%s" % (module, suffix), value)
        elif gt == GGML_Q8_0:
            # A selector codebook: dequantised to BF16.
            put(key, ("BF16", logical_shape(ne), bf16_from_q8_0(buf, data_start, ne, off)))
        else:
            # A norm or a conv base kernel: BF16.
            put(key, ("BF16", logical_shape(ne), bf16_from_float(buf, data_start, ne, gt, off)))

    os.makedirs(args.out_dir, exist_ok=True)
    st_path = os.path.join(args.out_dir, "model.safetensors")
    write_safetensors(st_path, out)

    with open(st_path, "rb") as f:
        out_sha = hashlib.sha256(f.read()).hexdigest()
    out_bytes = os.path.getsize(st_path)
    print("source_sha256=%s" % src_sha)
    print("tensors=%d" % len(out))
    print("output=%s" % st_path)
    print("output_bytes=%d" % out_bytes)
    print("output_sha256=%s" % out_sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
