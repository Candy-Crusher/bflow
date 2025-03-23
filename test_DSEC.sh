export HYDRA_FULL_ERROR=1
# Set DATA_DIR as the path to the DSEC dataset (parent of train and test dir)
DATA_DIR=./
# Set
# MDL_CFG=E_I_LU4_BD2_lowpyramid to use both events and frames, or
# MDL_CFG=E_LU4_BD2_lowpyramid to use only events
# MDL_CFG=E_I_LU4_BD2_lowpyramid
MDL_CFG=E_LU4_BD2_lowpyramid
# Set LOG_ONLY_NUMBERS=true to avoid logging images (can require a lot of space). Set to false by default.
LOG_ONLY_NUMBERS=true
CKPT=./E_LU4_BD2.ckpt

GPU_ID=0
python val.py model=raft-spline dataset=dsec dataset.path=${DATA_DIR} hardware.gpus=${GPU_ID} \
+experiment/dsec/raft_spline=${MDL_CFG} checkpoint=${CKPT}