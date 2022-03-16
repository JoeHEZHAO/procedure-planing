#!/bin/bash
PART=$1
NUM_PARTS=$2
CT_TFREC="/local/data0/p3iv/datasets/NIV_assets"
CT_LMDB="/local/data0/p3iv/datasets/COIN_assets/processed_data"

python3 encode_niv.py --part $PART --num_parts $NUM_PARTS --source $CT_TFREC --dest $CT_LMDB --dataset CrossTask