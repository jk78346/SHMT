for data in 'ls ../data/*.mp4'
do
	./gl2pose_estimation_3d -v $data > record.txt 
done
