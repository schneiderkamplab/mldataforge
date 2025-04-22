import filecmp
from mldataforge.commands.join import join_jsonl, join_mds, join_parquet
from mldataforge.utils import load_mds_directories, save_mds
import pytest

@pytest.mark.dependency(depends=["conversion"])
@pytest.mark.dependency(name="trafos")
@pytest.mark.parametrize("fmt,trafo,out_file,in_file", [
    ("jsonl", "from mldataforge.trafos import identity as process", "test.identity.jsonl", "test.jsonl"),
    ("jsonl", "from mldataforge.trafos import flatten_json as process", "test.flattened.jsonl", "test.jsonl"),
    ("jsonl", "from mldataforge.trafos import unflatten_json as process", "test.unflattened.jsonl", "test.flattened.jsonl"),
    ("mds", "from mldataforge.trafos import identity as process", "test.identity.mds", "test.jsonl.mds"),
    ("mds", "from mldataforge.trafos import flatten_json as process", "test.flattened.mds", "test.jsonl.mds"),
    ("mds", "from mldataforge.trafos import unflatten_json as process", "test.unflattened.mds", "test.flattened.mds"),
    ("parquet", "from mldataforge.trafos import identity as process", "test.identity.parquet", "test.jsonl.parquet"),
    ("parquet", "from mldataforge.trafos import flatten_json as process", "test.flattened.parquet", "test.jsonl.parquet"),
    ("parquet", "from mldataforge.trafos import unflatten_json as process", "test.unflattened.parquet", "test.flattened.parquet"),
])
def test_trafos(fmt, trafo, out_file, in_file, tmp_dir, scale_factor):
    if fmt == "jsonl":
        join_jsonl(
            output_file=str(tmp_dir / out_file),
            jsonl_files=[str(tmp_dir / in_file)],
            compression="infer",
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            trafo=trafo,
        )
        if "unflatten_json" in trafo or "idenitity" in trafo:
            assert filecmp.cmp(str(tmp_dir / out_file), str(tmp_dir / "test.jsonl"), shallow=False), f"Files {out_file} and test.jsonl are different"
    elif fmt == "mds":
        join_mds(
            output_dir=str(tmp_dir / out_file),
            mds_directories=[str(tmp_dir / in_file)],
            compression=None,
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            batch_size=2**10*scale_factor,
            buf_size=2**14*scale_factor,
            no_bulk=False,
            shard_size=2**26,
            no_pigz=True,
            trafo=trafo,
            shuffle=None,
            index=None,
        )
        if "unflatten_json" in trafo or "idenitity" in trafo:
            dircmp = filecmp.dircmp(str(tmp_dir / out_file), str(tmp_dir / "test.jsonl.mds"))
            assert len(dircmp.left_only) == 0, f"Left only files: {dircmp.left_only}"
            assert len(dircmp.right_only) == 0, f"Right only files: {dircmp.right_only}"
            assert len(dircmp.diff_files) == 0, f"Different files: {dircmp.diff_files}"
            assert len(dircmp.funny_files) == 0, f"Funny files: {dircmp.funny_files}"
    elif fmt == "parquet":
        join_parquet(
            output_file=str(tmp_dir / out_file),
            parquet_files=[str(tmp_dir / in_file)],
            compression="snappy",
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            batch_size=2**16,
            trafo=trafo,
        )
        if "unflatten_json" in trafo or "idenitity" in trafo:
            assert filecmp.cmp(str(tmp_dir / out_file), str(tmp_dir / "test.jsonl.parquet"), shallow=False), f"Files {out_file} and test.jsonl.parquet are different"
"""
compression=None, compression_args={"processes": 64}, buf_size=2**24, pigz=True, shard_size=None, size_hint=None, overwrite=True, yes=True, trafo=None):
"""
@pytest.mark.dependency(name="conversion")
@pytest.mark.dependency(name="tokenize")
def test_tokenize(tmp_dir):
    save_mds(
        load_mds_directories([str(tmp_dir / "test.jsonl.mds")]),
        output_dir=str(tmp_dir / "test.tokenized.mds"),
        overwrite=True,
        yes=True,
        trafo="""from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")
def process(sample):
    sample["text"] = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
    sample["input_ids"] = tokenizer(sample["text"], return_tensors="np")["input_ids"]
    return sample
""",
    )