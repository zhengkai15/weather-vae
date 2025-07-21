# !/bin/bash
set -x
PY=python
cd weather-vae
SCRIPT_NAME=main.py
${PY} -u ${SCRIPT_NAME} >> ${SCRIPT_NAME}.log