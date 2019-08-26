#!/bin/bash




# NConv
#python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth -network_file network_exp_unguided_depth -params_sub_dir Experiments/Old
#python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth -network_file network_exp_unguided_depth -params_sub_dir Experiments/Default
#python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth -network_file network_exp_unguided_depth -params_sub_dir Experiments/Default2
#python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth -network_file network_exp_unguided_depth -params_sub_dir Experiments/Default3
#python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth -network_file network_exp_unguided_depth -params_sub_dir Experiments/Default4

# Unguided_d
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Old
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Deconv
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/NoBias
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Stride
#
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Default2
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Deconv2
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/NoBias2
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Stride2
#
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Default3
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Deconv3
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/NoBias3
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Stride3
#
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Default4
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Deconv4
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/NoBias4
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Stride4
#
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Default_cb
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Default_cb2
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Default_cb3
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Default_cb4
#
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier2
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier3
#python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier4

python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/XavierConstBias
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/XavierConstBias2
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/XavierConstBias3
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/XavierConstBias4

python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/XavierNoBias
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/XavierNoBias1
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/XavierNoBias2
python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/XavierNoBias3


## Unguided_dg
#python run_nconv_cnn.py -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg -params_sub_dir Experiments/Default
#python run_nconv_cnn.py -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg -params_sub_dir Experiments/Default2
#python run_nconv_cnn.py -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg -params_sub_dir Experiments/Default3
#python run_nconv_cnn.py -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg -params_sub_dir Experiments/Default4

#python run_nconv_cnn.py -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg -params_sub_dir Experiments/NoBias_d
#python run_nconv_cnn.py -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg -params_sub_dir Experiments/NoBias_d2
#python run_nconv_cnn.py -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg -params_sub_dir Experiments/NoBias_d3
#python run_nconv_cnn.py -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg -params_sub_dir Experiments/NoBias_d4


# Unguided_ds
#python run_nconv_cnn.py -mode traineval -network_file Unguided_ds -ws_path workspace/StructNConv/Unguided_ds -params_sub_dir Experiments/Default