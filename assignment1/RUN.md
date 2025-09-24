# export DISPLAY=host.docker.internal:0.0
# Q1
## run 3 images provided by course staff
bash ./run_q1.sh
## run 2 images I captured
##  python affine_rectify.py \
##  --img data/q1/door2.jpg  \
##  --interactive \
##  --outdir out_q1_door2 \
##  --save-ann data/annotation/user_door2.npy #use UI to annotate

python affine_rectify.py \
--img data/q1/door2.jpg \
--outdir out_q1_door2 \
--ann ./data/annotation/user_door2.npy \
--key door2

##  python affine_rectify.py \
##  --img data/q1/floor1.jpg \
##  --interactive \
##  --outdir out_q1_floor1 \
##  --save-ann data/annotation/user_floor1.npy

python affine_rectify.py \
--img data/q1/floor1.jpg \
--outdir out_q1_floor1 \
--ann ./data/annotation/user_floor1.npy \
--key floor1

# Q2
## run 3 images provided by course staff
bash ./run_q2.sh
## run 2 images I captured
python affine_rectify.py \
--img data/q1/door2.jpg \
--outdir out_q2/door2/affine \
--ann ./data/annotation/user_door2.npy \
--key door2

## python metric_rectify_from_affine.py \
## --image data/q1/door2.jpg \
## --interactive \
## --key door2 \
## --H_aff out_q2/door2/affine/H_aff.npy \
## --save-ann data/annotation/user_q2_door2.npy \
## --out out_q2/door2/metric

python metric_rectify_from_affine.py \
--image data/q1/door2.jpg \
--ann data/annotation/user_q2_door2.npy \
--key door2 \
--H_aff out_q2/door2/affine/H_aff.npy \
--out out_q2/door2/metric

python affine_rectify.py \
--img data/q1/floor1.jpg \
--outdir out_q2/floor1/affine \
--ann ./data/annotation/user_floor1.npy \
--key floor1

## python metric_rectify_from_affine.py \
## --image data/q1/floor1.jpg \
## --interactive \
## --key floor1 \
## --H_aff out_q2/floor1/affine/H_aff.npy \
## --save-ann data/annotation/user_q2_floor1.npy \
## --out out_q2/floor1/metric

python metric_rectify_from_affine.py \
--image data/q1/floor1.jpg \
--ann data/annotation/user_q2_floor1.npy \
--key floor1 \
--H_aff out_q2/floor1/affine/H_aff.npy \
--out out_q2/floor1/metric

