#!/bin/bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
python -m unittest discover -s test -p "test_*.py"
