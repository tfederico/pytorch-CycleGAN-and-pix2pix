from os import listdir
from os.path import isfile, join
import csv

base = "data/SynthHands_Release/"
l1_dirs = ["female_noobject", "male_noobject"]
l2_dirs = ["seq0" + str(i) for i in range(1,8)]
l3_dirs = ["cam0" + str(i) for i in range(1,6)]
l4_dirs = ["0" + str(i) for i in range(1,4)]

paths = []
files = []

n_dirs = len(l1_dirs)*len(l2_dirs)*len(l3_dirs)*len(l4_dirs)
scanned = 0

for i in l1_dirs:
	for j in l2_dirs:
		for k in l3_dirs:
			for l in l4_dirs:
				mypath = base+i+"/"+j+"/"+k+"/"+l+"/"
				onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
				paths.append(mypath)
				files.append(len(onlyfiles)//4)
				scanned += 1
				print(str(100*scanned/n_dirs)+"% of directories parsed")

with open('parsed_dirs.csv', mode='w') as out:
	out_writer = csv.writer(out, delimiter=',')
	for i in range(len(paths)):
		out_writer.writerow([paths[i], files[i]])
