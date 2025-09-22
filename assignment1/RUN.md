# export DISPLAY=host.docker.internal:0.0
# Q1
# run 3 images provided by course staff
bash ./run_q1.sh
# run 2 images I captured
# python affine_rectify.py   --img data/q1/door2.jpg   --interactive   --outdir out_q1_door2   --save-ann data/annotation/user_door2.npy #use UI to annotate
python affine_rectify.py   --img data/q1/door2.jpg --outdir out_q1_door2 --ann ./data/annotation/user_door2.npy --key door2
# python affine_rectify.py   --img data/q1/floor1.jpg   --interactive   --outdir out_q1_floor1   --save-ann data/annotation/user_floor1.npy
python affine_rectify.py   --img data/q1/floor1.jpg --outdir out_q1_floor1 --ann ./data/annotation/user_floor1.npy --key floor1
