#! /bin/sh

for f in train/*/*.wav
do
	./SMILExtract -C openSMILE-2.2rc1/config/PLP_E_D_A.conf -I "$f" -outputcsv "${f%.*}".csv
done
for f in test/*/*.wav
do
	./SMILExtract -C openSMILE-2.2rc1/config/PLP_E_D_A.conf -I "$f" -outputcsv "${f%.*}".csv
done
