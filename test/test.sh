#!/bin/bash
pytest -vv -rs --durations=1000 -x --tmp-path test/tmp --cache-clear $@
