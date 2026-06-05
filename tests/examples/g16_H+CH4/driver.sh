#!/bin/bash

export PYTHONBUFFERED=1
export GAUSS_SCRDIR=/tmp/GAU-pyrinst

pyrinst-driver --parallel -P Gaussian --working-dir beads -F header_doublet --runcmd g16
