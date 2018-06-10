#! /bin/sh
for dir in ./test/*
do
		dir=${dir%*/}
		echo $dir
		for f in $dir/*.wav
		do
			SMILExtract -C openSMILE-2.2rc1/config/emobase2010.conf -I "$f" -outputcsv "${f%.*}".csv
    done
done
