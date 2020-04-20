##########
# example: ./Test.sh -m moving.nii.gz -l moving_label.nii.gz -n M015_00 
Continue=False
fixed=../data/MAPS_Dataset_Affined/sw_M023-1_12_tissue_rigidAlign.img
while getopts "m:l:n:c:f:" OPT
        do
        case $OPT in
                m)
                moving=$OPTARG
                ;;
                l)
                moving_label=$OPTARG
                ;;
		n)
		moving_name=$OPTARG
		;;
		c)
		Continue=$OPTARG
		;;
		f)
		fixed=$OPTARG
		;;

        esac
        done
CUDA_id=2
compose=../Compose/CompositITKDeformationField
iter=1500
SAVE_DIR=../data/results
file=$SAVE_DIR
iter_num=5
if [ $Continue == "False" ];
then
	for i in {1..1};
	do 
		echo $moving
		CUDA_VISIBLE_DEVICES=$CUDA_id python test.py $CUDA_id ../models $iter $fixed $moving $moving_label
		echo "[Stage ${i}]: Warping moving image Finished!" >> log.md
		python dice.py $SAVE_DIR/warped_seg.nii.gz $fixed 150 >> log.md
        	python dice.py $SAVE_DIR/warped_seg.nii.gz $fixed 250 >> log.md
		mv $SAVE_DIR/field.mha $SAVE_DIR/field_${i}.mha
		mv $SAVE_DIR/log.mha $SAVE_DIR/log_${i}.mha
		cp $SAVE_DIR/field_${i}.mha $SAVE_DIR/compose_0.mha
		python Compute_J_negative.py $SAVE_DIR/field_${i}.mha $moving
	done

	for i in {2..5}
	do

		CUDA_VISIBLE_DEVICES=$CUDA_id python test.py $CUDA_id ../models $iter $fixed $SAVE_DIR/warped_seg.nii.gz $moving_label
		mv $SAVE_DIR/field.mha $SAVE_DIR/field_${i}.mha
		mv $SAVE_DIR/log.mha $SAVE_DIR/log_${i}.mha
		$compose $SAVE_DIR/compose_$((i-2)).mha $SAVE_DIR/field_${i}.mha $SAVE_DIR/compose_$((i-1)).mha
		CUDA_VISIBLE_DEVICES=$CUDA_id python DNN_Flow_Warped.py $CUDA_id $moving $SAVE_DIR/compose_$((i-1)).mha
		echo "[Stage ${i}]: Warping moving image Finished!" >> log.md
		python dice.py $SAVE_DIR/warped_seg.nii.gz $fixed 150 >> log.md
        	python dice.py $SAVE_DIR/warped_seg.nii.gz $fixed 250 >> log.md
		python Compute_J_negative.py $SAVE_DIR/compose_$((i-1)).mha $moving
	done
	cp $SAVE_DIR/compose_$((i-1)).mha $SAVE_DIR/compose.mha
fi

python dice.py $SAVE_DIR/warped_seg.nii.gz $fixed 150
python dice.py $SAVE_DIR/warped_seg.nii.gz $fixed 250
echo "Dice computation finished!"
python Compute_J_negative.py $SAVE_DIR/compose.mha $moving
echo "Negative Jac finished!"

for i in {1..5}
do
	echo "[Stage $i]: RFP (%)" >> log.md
	if [ $i == 1 ];
	then
		
		python Compute_J_negative.py $file/field_${i}.mha $fixed >> log.md
	else
		python Compute_J_negative.py $file/compose_$((i-1)).mha $fixed >> log.md
	fi
done

# if you want save space, pleace execute the following command
#rm $file/field*
#rm $file/compose_*

