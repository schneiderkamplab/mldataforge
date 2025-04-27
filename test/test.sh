#!/bin/bash
pytest -vv -rs --durations=5 -x --tmp-path test/tmp --cache-clear $@
