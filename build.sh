#!/bin/bash
set -e

# pyHalo is not on PyPI, install from GitHub
pip install git+https://github.com/dangilman/pyHalo.git

# DeepLenseSim is already in the repo
pip install ./DeepLenseSim

# Standard dependencies
pip install -r requirements.txt
