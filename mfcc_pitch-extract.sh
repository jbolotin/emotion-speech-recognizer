#! /bin/sh

for f in train/*/*.wav
do
	./SMILExtract -C mfcc_pitch.conf -I "$f" -outputcsv "${f%.*}".csv
done
for f in test/*/*.wav
do
	./SMILExtract -C mfcc_pitch.conf -I "$f" -outputcsv "${f%.*}".csv
done
