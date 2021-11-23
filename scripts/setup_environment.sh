#!/bin/bash

export PYTHONPATH=${PWD}:$PYTHONPATH
workdir=$PWD
python --version
pip install virtualenv
rm -rf clothing_classifier_env
virtualenv clothing_classifier_env
source clothing_classifier_env/bin/activate
pip install -r requirements.txt


