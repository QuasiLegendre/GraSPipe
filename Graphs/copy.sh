mkdir Generated
for i in {25427..25456}
do
	for j in {1..10}
	do
		cp ./Gen/sub-00${i}_ses-${j}_dwi_desikan_space-MNI152NLin6_res-2x2x2_measure-spatial-ds_adj.ssv ./Generated/${i}_${j}.ssv
	done
done
