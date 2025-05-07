# mldataforge
swiss army knife of scripts for transforming and processing datasets for machine learning

## conversion
Currently, mldataforge provides space- and time-efficient conversions between JSONL (with or without compression), MosaiclML Dataset (MDS format), and Parquet. The implementations handle conversions by individual samples or small batches of samples and make efficient use of multi-core architectures where possible. Consequently, mldataforge is an excellent choice when transforming TB-scale datasets on data processing nodes with many cores.

## splitting
Currently, mldataforge provides space- and time-efficient splitting of JSONL (with or without compression). The implementations handle conversions by individual samples or small batches of samples and make efficient use of multi-core architectures where possible. The splitting function can take an already splitted dataset and re-split it with a different granularity.

## installation and general usage
```
pip install mldataforge
python -m mldataforge --help
```

## usage example: converting MosaiclML Dataset (MDS) to Parquet format
```
Usage: python -m mldataforge convert mds parquet [OPTIONS] OUTPUT_FILE
                                                 MDS_DIRECTORIES...

Options:
  --compression [snappy|gzip|zstd]
                                  Compress the output file (default: snappy).
  --overwrite                     Overwrite existing path.
  --yes                           Assume yes to all prompts. Use with caution
                                  as it will remove files or even entire
                                  directories without confirmation.
  --batch-size INTEGER            Batch size for loading data and writing
                                  files (default: 65536).
  --no-bulk                       Use a custom space and time-efficient bulk
                                  reader (only gzip and no compression).
  --help                          Show this message and exit.
```