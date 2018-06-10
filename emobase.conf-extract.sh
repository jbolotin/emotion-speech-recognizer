#! /bin/sh
for dir in ./train/*
do
		dir=${dir%*/}
		echo $dir
		for f in $dir/*.wav
		do
			SMILExtract -C openSMILE-2.2rc1/config/emobase.conf -I "$f" -O emobase.conf-train.arff -instname $dir
    done
done
for dir in ./test/*
do
		dir=${dir%*/}
		echo $dir
		for f in $dir/*.wav
		do
			SMILExtract -C openSMILE-2.2rc1/config/emobase.conf -I "$f" -O emobase.conf-test.arff -instname $dir
    done
done
