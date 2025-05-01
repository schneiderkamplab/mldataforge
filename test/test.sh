#!/bin/bash
pytest -vv -rs --durations=10 -x --tmp-path test/tmp --cache-clear $@
