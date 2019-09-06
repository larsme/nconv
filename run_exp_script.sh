#!/bin/bash

## parameter counts
#python run_nconv_cnn.py -mode count_parameters -ws_path workspace/exp_unguided_depth -network_file network_exp_unguided_depth -params_sub_dir Experiments/Default
#
#python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_d -network_file Unguided_d -params_sub_dir Experiments/Kaiming
#python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_d -network_file Unguided_d -params_sub_dir Experiments/Xavier_no_bias
#python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_d -network_file Unguided_d -params_sub_dir Experiments/Xavier_no_bias_deconv
#python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_d -network_file Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride
#python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_d -network_file Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride_deconv
#
#python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_d -network_file Unguided_d -params_sub_dir Experiments5channels/Default
#python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_d -network_file Unguided_d -params_sub_dir Experiments5channels/Deconv
#python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_d -network_file Unguided_d -params_sub_dir Experiments5channels/NoBias
#python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_d -network_file Unguided_d -params_sub_dir Experiments5channels/Stride
#
#python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_dg -network_file Unguided_dg -params_sub_dir Experiments/Default
#python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_ds -network_file Unguided_ds -params_sub_dir Experiments/Default
#python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_dsg -network_file Unguided_dsg -params_sub_dir Experiments/Default


# NConv
#python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth -network_file network_exp_unguided_depth -params_sub_dir Experiments/Old
#python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth -network_file network_exp_unguided_depth -params_sub_dir Experiments/Default
#python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth -network_file network_exp_unguided_depth -params_sub_dir Experiments/Default2
#python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth -network_file network_exp_unguided_depth -params_sub_dir Experiments/Default3
#python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth -network_file network_exp_unguided_depth -params_sub_dir Experiments/Default4

# Unguided_d
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Kaiming -exp_subdir 1
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Kaiming -exp_subdir 2
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Kaiming -exp_subdir 3
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Kaiming -exp_subdir 4

python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Kaiming_const_bias -exp_subdir 1
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Kaiming_const_bias -exp_subdir 2
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Kaiming_const_bias -exp_subdir 3
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Kaiming_const_bias -exp_subdir 4

python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier -exp_subdir 1
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier -exp_subdir 2
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier -exp_subdir 3
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier -exp_subdir 4

python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_const_bias -exp_subdir 1
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_const_bias -exp_subdir 2
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_const_bias -exp_subdir 3
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_const_bias -exp_subdir 4

python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias -exp_subdir 1
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias -exp_subdir 2
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias -exp_subdir 3
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias -exp_subdir 4

python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_lidar_pad -exp_subdir 1
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_lidar_pad -exp_subdir 2
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_lidar_pad -exp_subdir 3
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_lidar_pad -exp_subdir 4

python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_deconv -exp_subdir 1
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_deconv -exp_subdir 2
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_deconv -exp_subdir 3
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_deconv -exp_subdir 4

python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride -exp_subdir 1
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride -exp_subdir 2
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride -exp_subdir 3
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride -exp_subdir 4

python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride_deconv -exp_subdir 1
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride_deconv -exp_subdir 2
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride_deconv -exp_subdir 3
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride_deconv -exp_subdir 4


python run_nconv_cnn.py  -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg -params_sub_dir Experiments/Default -exp_subdir 1
python run_nconv_cnn.py  -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg -params_sub_dir Experiments/Default -exp_subdir 2
python run_nconv_cnn.py  -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg -params_sub_dir Experiments/Default -exp_subdir 3
python run_nconv_cnn.py  -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg -params_sub_dir Experiments/Default -exp_subdir 4


python run_nconv_cnn.py  -mode traineval -network_file Unguided_ds -ws_path workspace/StructNConv/Unguided_ds -params_sub_dir Experiments/Default -exp_subdir 1
python run_nconv_cnn.py  -mode traineval -network_file Unguided_ds -ws_path workspace/StructNConv/Unguided_ds -params_sub_dir Experiments/Default -exp_subdir 2
python run_nconv_cnn.py  -mode traineval -network_file Unguided_ds -ws_path workspace/StructNConv/Unguided_ds -params_sub_dir Experiments/Default -exp_subdir 3
python run_nconv_cnn.py  -mode traineval -network_file Unguided_ds -ws_path workspace/StructNConv/Unguided_ds -params_sub_dir Experiments/Default -exp_subdir 4


