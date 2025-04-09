#!/bin/bash
DIR=$(realpath $(dirname ${BASH_SOURCE[0]}))
echo convert parquet to jsonl
python -m mldataforge convert parquet jsonl $DIR/test.parquet.jsonl.gz $DIR/test.parquet --overwrite --yes
echo convert jsonl to parquet
python -m mldataforge convert jsonl parquet $DIR/test.parquet.jsonl.parquet $DIR/test.parquet.jsonl.gz --overwrite --yes
echo convert parquet to mds
python -m mldataforge convert parquet mds $DIR/test.parquet.mds $DIR/test.parquet --overwrite --yes
echo convert jsonl to mds
python -m mldataforge convert jsonl mds $DIR/test.parquet.jsonl.mds $DIR/test.parquet.jsonl.gz --overwrite --yes
echo convert mds to parquet
python -m mldataforge convert mds parquet $DIR/test.parquet.mds.parquet $DIR/test.parquet.mds --overwrite --yes
echo convert mds to jsonl
python -m mldataforge convert mds jsonl $DIR/test.parquet.mds.jsonl.gz $DIR/test.parquet.mds --overwrite --yes
