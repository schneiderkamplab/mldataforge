import filecmp
from mldataforge.commands.convert.jinx import jinx_to_jsonl
from mldataforge.commands.join import join_jinx
import pytest

@pytest.mark.parametrize("encoding", [
    pytest.param("a85", marks=pytest.mark.dependency(name="encoding_a85", dependency=["convert_jsonl_jinx"], scope="session")),
    pytest.param("b64", marks=pytest.mark.dependency(name="encoding_b64", dependency=["convert_jsonl_jinx"], scope="session")),
    pytest.param("bin", marks=pytest.mark.dependency(name="encoding_bin", dependency=["convert_jsonl_jinx"], scope="session")),
])
def test_encoding(encoding, tmp_dir):
    input_file = str(tmp_dir / "test.jsonl.jinx")
    output_file = str(tmp_dir / f"test.{encoding}.jinx")
    if encoding in ["a85", "b64"]:
        join_jinx(
            output_file=output_file,
            jinx_paths=[input_file],
            compression="zstd",
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            shard_size=None,
            trafo=None,
            mmap=False,
            shuffle=None,
            index=None,
            sort_key=None,
            lazy=False,
            compress_threshold=2**6,
            compress_ratio=1.0,
            encoding=encoding,
            binary_threshold=None,
            ext_sep=".",
            override_encoding=None,
        )
    elif encoding == "bin":
        join_jinx(
            output_file=output_file,
            jinx_paths=[input_file],
            compression="zstd",
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            shard_size=None,
            trafo=None,
            mmap=False,
            shuffle=None,
            index=None,
            sort_key=None,
            lazy=False,
            compress_threshold=2**6,
            compress_ratio=1.0,
            encoding=None,
            binary_threshold=2**3,
            ext_sep=".",
            override_encoding=None,
        )

@pytest.mark.parametrize("encoding", [
    pytest.param("a85", marks=pytest.mark.dependency(dependency=["encoding_a85"], scope="session")),
    pytest.param("b64", marks=pytest.mark.dependency(dependency=["encoding_b64"], scope="session")),
    pytest.param("bin", marks=pytest.mark.dependency(dependency=["encoding_bin"], scope="session")),
])
def test_decoding(encoding, tmp_dir, jsonl_tools):
    input_file = str(tmp_dir / f"test.{encoding}.jinx")
    output_file = str(tmp_dir / f"test.{encoding}.jsonl")
    jinx_to_jsonl(
        output_file=output_file,
        jinx_paths=[input_file],
        compression=None,
        compression_args={"processes": 64},
        overwrite=True,
        yes=True,
        trafo=None,
        mmap=False,
        split=None,
        shuffle=None,
        index=None,
        sort_key=None,
        lazy=False,
        override_encoding=None,
    )
    assert jsonl_tools.equal(str(tmp_dir / "test.jsonl"), output_file), f"Files {output_file} and test.jsonl are different"
