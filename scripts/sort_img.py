import os
import sys
import shutil

train_csv = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]
count = {}

f = open(train_csv,"r")
skip=1
for l in f.readlines():
   if skip>0:
      skip=skip-1
      continue
   (img_file,category) = l.strip().split(",")
   img_file=img_file.strip()
   category=category.strip()
   if category not in count:count[category]=0
   cat_output_dir = os.path.join(output_dir,category)
   if os.path.exists(cat_output_dir)==False:
      os.makedirs(cat_output_dir)
   input_file = os.path.join(input_dir,img_file)
   output_file = os.path.join(cat_output_dir,img_file)
   shutil.copyfile(input_file,output_file)
   count[category]=count[category]+1
   print("%s -> %s"%(img_file,category))
f.close()