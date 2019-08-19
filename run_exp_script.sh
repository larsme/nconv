#!/bin/bash

python run_nconv_cnn.py -mode train -exp Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Default
python run_nconv_cnn.py -mode train -exp Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Deconv
python run_nconv_cnn.py -mode train -exp Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir NoBias
python run_nconv_cnn.py -mode train -exp Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Stride


