#!/bin/bash
pytest -vv -rs --durations=5 -x --tmp-path test/tmp --samples 10000
