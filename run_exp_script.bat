::!/bin/bash

::python run_nconv_cnn.py -mode count_parameters -ws_path workspace/exp_unguided_depth -network_file network_exp_unguided_depth -params_sub_dir Experiments/Default -exp_subdir 1
::python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth -network_file network_exp_unguided_depth -params_sub_dir Experiments/Default
::python run_nconv_cnn.py -mode display -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90 -exp_subdir 1
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90 -exp_subdir 1
::python run_nconv_cnn.py -mode traineval -network_file network_exp_guided_nconv_cnn_l1 -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_rotate_90_guided -exp_subdir 1
::python run_nconv_cnn.py -mode display -network_file network_exp_guided_nconv_cnn_l1 -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90_guided -exp_subdir 1
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias -exp_subdir 1
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Kaiming -exp_subdir 1




:::: parameter counts
::python run_nconv_cnn.py -mode count_parameters -ws_path workspace/exp_unguided_depth -network_file network_exp_unguided_depth -params_sub_dir Experiments/Default
::
::python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_d -network_file Unguided_d -params_sub_dir Experiments/Kaiming
::python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_d -network_file Unguided_d -params_sub_dir Experiments/Xavier_no_bias
::python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_d -network_file Unguided_d -params_sub_dir Experiments/Xavier_no_bias_deconv
::python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_d -network_file Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride
::python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_d -network_file Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride_deconv
::
::python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_d -network_file Unguided_d -params_sub_dir Experiments5channels/Default
::python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_d -network_file Unguided_d -params_sub_dir Experiments5channels/Deconv
::python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_d -network_file Unguided_d -params_sub_dir Experiments5channels/NoBias
::python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_d -network_file Unguided_d -params_sub_dir Experiments5channels/Stride
::
::python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_dg -network_file Unguided_dg -params_sub_dir Experiments/Default
::python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_ds -network_file Unguided_ds -params_sub_dir Experiments/Default
::python run_nconv_cnn.py -mode count_parameters -ws_path workspace/StructNConv/Unguided_dsg -network_file Unguided_dsg -params_sub_dir Experiments/Default


:: NConv
::python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth -network_file network_exp_unguided_depth -params_sub_dir Experiments/Old
::python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth -network_file network_exp_unguided_depth -params_sub_dir Experiments/Default
::python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth -network_file network_exp_unguided_depth -params_sub_dir Experiments/Default2
::python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth -network_file network_exp_unguided_depth -params_sub_dir Experiments/Default3
::python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth -network_file network_exp_unguided_depth -params_sub_dir Experiments/Default4

::python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth/Experiments/Xavier -network_file network_exp_unguided_depth -exp_subdir 1
::python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth/Experiments/Xavier -network_file network_exp_unguided_depth -exp_subdir 2
::python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth/Experiments/Xavier -network_file network_exp_unguided_depth -exp_subdir 3
::python run_nconv_cnn.py -mode traineval -ws_path workspace/exp_unguided_depth/Experiments/Xavier -network_file network_exp_unguided_depth -exp_subdir 4

:: Unguided_d
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Kaiming -exp_subdir 1
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Kaiming -exp_subdir 2
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Kaiming -exp_subdir 3
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Kaiming -exp_subdir 4

::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Kaiming_const_bias -exp_subdir 1
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Kaiming_const_bias -exp_subdir 2
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Kaiming_const_bias -exp_subdir 3
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Kaiming_const_bias -exp_subdir 4

::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier -exp_subdir 1
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier -exp_subdir 2
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier -exp_subdir 3
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier -exp_subdir 4

::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_const_bias -exp_subdir 1
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_const_bias -exp_subdir 2
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_const_bias -exp_subdir 3
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_const_bias -exp_subdir 4

::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias -exp_subdir 1
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias -exp_subdir 2
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias -exp_subdir 3
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias -exp_subdir 4

::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_lidar_pad -exp_subdir 1
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_lidar_pad -exp_subdir 2
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_lidar_pad -exp_subdir 3
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_lidar_pad -exp_subdir 4

::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_deconv -exp_subdir 1
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_deconv -exp_subdir 2
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_deconv -exp_subdir 3
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_deconv -exp_subdir 4

::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride -exp_subdir 1
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride -exp_subdir 2
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride -exp_subdir 3
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride -exp_subdir 4

::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride_deconv -exp_subdir 1
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride_deconv -exp_subdir 2
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride_deconv -exp_subdir 3
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride_deconv -exp_subdir 4

::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_4_channels -exp_subdir 1
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_4_channels -exp_subdir 2
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_4_channels -exp_subdir 3
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_4_channels -exp_subdir 4


::
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_deconv_no_devalue -exp_subdir 1
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_deconv_no_devalue -exp_subdir 2
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_deconv_no_devalue -exp_subdir 3
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_deconv_no_devalue -exp_subdir 4

::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride_no_devalue -exp_subdir 1
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride_no_devalue -exp_subdir 2
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride_no_devalue -exp_subdir 3
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride_no_devalue -exp_subdir 4

::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride_deconv_no_devalue -exp_subdir 1
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride_deconv_no_devalue -exp_subdir 2
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride_deconv_no_devalue -exp_subdir 3
::python run_nconv_cnn.py -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_stride_deconv_no_devalue -exp_subdir 4


::python run_nconv_cnn.py  -mode traineval -network_file Unguided_ds -ws_path workspace/StructNConv/Unguided_ds -params_sub_dir Experiments/Default -exp_subdir 1

::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_mirrored -exp_subdir 1
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_mirrored -exp_subdir 2
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_no_bias_mirrored -exp_subdir 3

::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir Experiments/Xavier_nI_nb_m_1c_pS -exp_subdir 1

python run_nconv_cnn.py  -mode traineval -network_file Unguided_de -ws_path workspace/StructNConv/Unguided_de -params_sub_dir Experiments/simplify_e_from_d -exp_subdir 1 -evaluate_all_epochs true
python run_nconv_cnn.py  -mode traineval -network_file Unguided_de -ws_path workspace/StructNConv/Unguided_de -params_sub_dir Experiments/simplify_e_from_d -exp_subdir 2 -evaluate_all_epochs true
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_de2 -ws_path workspace/StructNConv/Unguided_de -params_sub_dir Experiments/sep_pow_smoothl1_rec -exp_subdir 2 -evaluate_all_epochs true




::python run_nconv_cnn.py  -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg -params_sub_dir Experiments/Default -exp_subdir 1
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg -params_sub_dir Experiments/Default -exp_subdir 2
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg -params_sub_dir Experiments/Default -exp_subdir 3
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg -params_sub_dir Experiments/Default -exp_subdir 4
::

::python run_nconv_cnn.py  -mode traineval -network_file Guided_ds_simple -ws_path workspace/StructNConv/Guided_ds_simple -params_sub_dir Experiments/Default -exp_subdir 1
::python run_nconv_cnn.py  -mode traineval -network_file Guided_ds_simple -ws_path workspace/StructNConv/Guided_ds_simple -params_sub_dir Experiments/Default -exp_subdir 2
::python run_nconv_cnn.py  -mode traineval -network_file Guided_ds_simple -ws_path workspace/StructNConv/Guided_ds_simple -params_sub_dir Experiments/Default -exp_subdir 3
::python run_nconv_cnn.py  -mode traineval -network_file Guided_ds_simple -ws_path workspace/StructNConv/Guided_ds_simple -params_sub_dir Experiments/Default -exp_subdir 4
::
::
::
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_ds_simple -ws_path workspace/StructNConv/Unguided_ds_simple -params_sub_dir Experiments/Default -exp_subdir 1
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_ds_simple -ws_path workspace/StructNConv/Unguided_ds_simple -params_sub_dir Experiments/Default -exp_subdir 2
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_ds_simple -ws_path workspace/StructNConv/Unguided_ds_simple -params_sub_dir Experiments/Default -exp_subdir 3
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_ds_simple -ws_path workspace/StructNConv/Unguided_ds_simple -params_sub_dir Experiments/Default -exp_subdir 4


::python run_nconv_cnn.py  -mode traineval -network_file Unguided_ds_simple -ws_path workspace/StructNConv/Unguided_ds_simple -params_sub_dir Experiments/Default -exp_subdir 1
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_ds_simple -ws_path workspace/StructNConv/Unguided_ds_simple -params_sub_dir Experiments/Default -exp_subdir 2
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_ds_simple -ws_path workspace/StructNConv/Unguided_ds_simple -params_sub_dir Experiments/Default -exp_subdir 3
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_ds_simple -ws_path workspace/StructNConv/Unguided_ds_simple -params_sub_dir Experiments/Default -exp_subdir 4

::python run_nconv_cnn.py  -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg_4_stage -params_sub_dir Experiments/Default -exp_subdir 1
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg_4_stage -params_sub_dir Experiments/Default -exp_subdir 2
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg_4_stage -params_sub_dir Experiments/Default -exp_subdir 3
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_dg -ws_path workspace/StructNConv/Unguided_dg_4_stage -params_sub_dir Experiments/Default -exp_subdir 4

::python run_nconv_cnn.py  -mode traineval -network_file Unguided_dsg -ws_path workspace/StructNConv/Unguided_dsg -params_sub_dir Experiments/Default -exp_subdir 1
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_dsg -ws_path workspace/StructNConv/Unguided_dsg -params_sub_dir Experiments/Default -exp_subdir 2
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_dsg -ws_path workspace/StructNConv/Unguided_dsg -params_sub_dir Experiments/Default -exp_subdir 3
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_dsg -ws_path workspace/StructNConv/Unguided_dsg -params_sub_dir Experiments/Default -exp_subdir 4










::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

:: Own dataset


::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_rotate_180 -exp_subdir 1
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_rotate_180 -exp_subdir 2
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_rotate_180 -exp_subdir 3
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_rotate_180 -exp_subdir 4
::
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias -exp_subdir 1
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias -exp_subdir 2
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias -exp_subdir 3
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias -exp_subdir 4
::

::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_rotate_90_rand -exp_subdir 1
::
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_rotate_90 -exp_subdir 1
::
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90 -exp_subdir 1
::
::python run_nconv_cnn.py  -mode traineval -network_file network_exp_guided_nconv_cnn_l1 -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90_guided -exp_subdir 1
::python run_nconv_cnn.py  -mode traineval -network_file network_exp_guided_nconv_cnn_l1 -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90_guided -exp_subdir 2
::python run_nconv_cnn.py  -mode traineval -network_file network_exp_guided_nconv_cnn_l1 -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90_guided -exp_subdir 3
::python run_nconv_cnn.py  -mode traineval -network_file network_exp_guided_nconv_cnn_l1 -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90_guided -exp_subdir 4
::
::
::
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90 -exp_subdir 1
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90 -exp_subdir 2
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90 -exp_subdir 3
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90 -exp_subdir 4
::
::
::python run_nconv_cnn.py  -mode eval -network_file network_exp_guided_nconv_cnn_l1 -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90_guided -exp_subdir 1 -checkpoint_num 6
::python run_nconv_cnn.py  -mode eval -network_file network_exp_guided_nconv_cnn_l1 -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90_guided -exp_subdir 2 -checkpoint_num 6
::python run_nconv_cnn.py  -mode eval -network_file network_exp_guided_nconv_cnn_l1 -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90_guided -exp_subdir 3 -checkpoint_num 6
::python run_nconv_cnn.py  -mode eval -network_file network_exp_guided_nconv_cnn_l1 -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90_guided -exp_subdir 4 -checkpoint_num 6

::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90 -exp_subdir 1
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90 -exp_subdir 2
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90 -exp_subdir 3
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_lidar_padding_rotate_90 -exp_subdir 4
::
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_rotate_90 -exp_subdir 1
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_rotate_90 -exp_subdir 2
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_rotate_90 -exp_subdir 3
::python run_nconv_cnn.py  -mode traineval -network_file Unguided_d -ws_path workspace/StructNConv/Unguided_d -params_sub_dir ExperimentsNewDataset/Xavier_no_bias_rotate_90 -exp_subdir 4

