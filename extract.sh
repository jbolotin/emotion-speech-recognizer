for dir in ./train/*
do
		dir=${dir%*/}
		for f in $dir/*
		do
			SMILExtract -C opensmile-2.3.0/config/ComParE_2016.conf -I "$f" -O outputfiles/ComParE_2016.conf.arff -instname "$f"
		done
done
