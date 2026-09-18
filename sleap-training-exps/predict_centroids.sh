#!/bin/bash

VIDEO=/camp/lab/windingm/home/shared/sleap/model-tests/comparison_frames_plain.mp4
MODELS=/camp/lab/windingm/home/shared/models/sideview/experiments
OUT=/camp/lab/windingm/home/shared/sleap/model-tests/predictions

mkdir -p "$OUT"

for MODEL in \
    centroid_baseline \
    centroid_fullres_sigma5 \
    centroid_halfres_sigma2p5 \
    centroid_halfres_body \
    centroid_fullres_body
do
    echo "Running $MODEL"

    sleap predict \
        -i "$VIDEO" \
        -m "$MODELS/$MODEL" \
        -o "$OUT/$MODEL.slp" \
        --peak_threshold 0.2 \
        --batch_size 8
done