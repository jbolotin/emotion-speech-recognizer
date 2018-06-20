#! /bin/sh

for f in train/*/*.wav
do
	./SMILExtract -C openSMILE-2.2rc1/config/demo/demo1_energy.conf -I "$f" -O "${f%.*}".csv
done
for f in test/*/*.wav
do
	./SMILExtract -C openSMILE-2.2rc1/config/demo/demo1_energy.conf -I "$f" -O "${f%.*}".csv
done
