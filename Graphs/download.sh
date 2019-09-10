for i in {27055..27102}
do
	for j in {1..1}
	do
		#wget -P ./SWU4-JHU -O ./SWU4-JHU/${i}_${j}.gpickle https://mrneurodata.s3.amazonaws.com/data/SWU4/ndmg_0-0-48/graphs/JHU/sub-00${i}_ses-${j}_dwi_JHU.gpickle
		#wget -P ./BNU1-desikan -O ./BNU1-desikan/${i}_${j}.gpickle https://mrneurodata.s3.amazonaws.com/data/BNU1/ndmg_0-0-48/graphs/desikan/sub-00${i}_ses-${j}_dwi_desikan.gpickle
		#wget -P ./SWU4-desikan -O ./SWU4-desikan/${i}_${j}.gpickle https://mrneurodata.s3.amazonaws.com/data/SWU4/ndmg_0-0-48/graphs/desikan/sub-00${i}_ses-${j}_dwi_desikan.gpickle
		wget -P ./BNU3-JHU -O ./BNU3-JHU/${i}_${j}.gpickle https://mrneurodata.s3.amazonaws.com/data/BNU3/ndmg_0-0-48/graphs/JHU/sub-00${i}_ses-${j}_dwi_JHU.gpickle
		#wget -P ./Talairach -O ./Talairach/${i}_${j}.ssv https://mrneurodata.s3.amazonaws.com/data/HNU1/ndmg_0-0-48/graphs/Talairach/sub-00${i}_ses-${j}_dwi_Talairach.ssv
	done
done
