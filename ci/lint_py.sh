#!/bin/bash

set -exuo pipefail

# Passing RUNNING_IN_CI as argument cause this script to run in CI-mode.
# Othwerwise local mode will be used
#
# CI-mode:     Any violation of the formatting rule will trigger error
# local mode:  Violations of formatting rules are auto corrected in-place
RUNNING_IN_CI=0
[[ ($# == 1) && ($1 == "RUNNING_IN_CI") ]] && RUNNING_IN_CI=1

# Generate a warning if we're running a different version of yapf then expected
expected_version="0.40.2"
current_version=$(yapf --version | awk '{print $2}')
if [ "$current_version" != "$expected_version" ]; then
    echo -e "\033[1;33mWarning: You are running yapf version $current_version, but expected version is $expected_version\033[0m"
fi

# Find PY files
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON_DIR=$SCRIPT_DIR/../python
PY_FILES=$( find $PYTHON_DIR -type f \( -iname "*.py" \))

# Run YAPF
YAPF_ARGS="--in-place" # Fix in-place if we're not in CI
[[ $RUNNING_IN_CI == 1 ]] && YAPF_ARGS="--quiet" # output nothing and set return value in CI
yapf $YAPF_ARGS $PY_FILES

# Run pep257 docstring linting according to the "ament" convention
AMENT_SELECT="D401,D402,D404,D406,D408,D410,D412,D413,D414,D416,D418"
AMENT_IGNORES='D100,D101,D102,D103,D104,D105,D106,D107,D203,D212,D404'
pydocstyle --ignore $AMENT_IGNORES --add-select=$AMENT_SELECT $PY_FILES
