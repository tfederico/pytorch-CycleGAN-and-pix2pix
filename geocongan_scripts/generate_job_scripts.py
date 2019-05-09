import pandas as pd
import csv

file_ext = ".png"

df = pd.read_csv("parsed_dirs.csv", sep=',', header=None)

l1_dirs = ["female_noobject", "male_noobject"]
l2_dirs = ["seq0" + str(i) for i in range(1,8)]
l3_dirs = ["cam0" + str(i) for i in range(1,6)]
l4_dirs = ["0" + str(i) for i in range(1,4)]


index = 0
files = []
new_files = []
s = 0
for _, row in df.iterrows():
	for file_num in range(row[1]):

		file_name = "00000000"+str(file_num)
		file_name = file_name[-8:]
		file_path = str(row[0])+file_name+"_color"+file_ext
		new_file_name = "00000000"+str(index)
		new_file_name = new_file_name[-8:]
		files.append(file_path)
		new_files.append(new_file_name)
		index += 1
		
	s += row[1]
	job_script = ["#!/bin/bash --login",
	"#$ -cwd",
	"",
	"# The 'myprog' below is serial hence no '-pe' option needed",
	"",
	"#$ -t 1-"+str(row[1]),
	"    # ...tell SGE that this is an array job, with \"tasks\" numbered from 1",
	"    #    to ...",
	"X_PARAM=( "+" ".join(["\""+f+"\"" for f in files])+" )",
	"Y_PARAM=( "+" ".join(["\""+f+"\"" for f in new_files])+" )",
	"",
	"# Bash arrays use zero-based indexing but you CAN'T use -t 0-9 above (0 is an invalid task id)",
	"INDEX=$((SGE_TASK_ID-1))",
	"conda activate opencv",
	"python preprocessing_synthhands.py -ip ${X_PARAM[$INDEX]} -cp . -rp data/SynthHands_Preprocessed/cumulative/ -rn ${Y_PARAM[$INDEX]}"]

	with open("generated_jobs/job_"+row[0].replace("/", "_")[:-1]+".txt", "w") as text_file:
		text_file.write("\n".join(job_script))
	
with open('mapping.csv', mode='w') as out:
	out_writer = csv.writer(out, delimiter=',')
	for i in range(len(files)):
		out_writer.writerow([files[i], new_files[i]])
