#! /bin/sh
for dir in ./train/*
do
		dir=${dir%*/}
		echo $dir
		for f in $dir/*.wav
		do
			SMILExtract -C openSMILE-2.2rc1/config/emo_large.conf -I "$f" -O emo_large-train.arff -instname $dir
    done
done
