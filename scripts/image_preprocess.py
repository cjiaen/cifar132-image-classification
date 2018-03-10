import os
from PIL import Image

def preprocess(img_path,img_id,label,output_dir):
    dir = os.path.join(output_dir,label)
    images = []
    img = Image.open(img_path)
    w,h = img.size
    d1 = {}
    ratio = float(w)/float(h)
    if w<h:
        d1[331]=(331,int(331/ratio))
        #d1[363]=(363,int(331/ratio))
        #d1[395]=(395,int(331/ratio))
        d1[427]=(427,int(427/ratio))
    else:
        d1[331]=(int(ratio*331),331)
        #d1[363]=(int(ratio*363),363)
        #d1[395]=(int(ratio*395),395)
        d1[427]=(int(ratio*427),427)
    sq={}
    if w<h:
        for sq_d in d1:
            sq_w,sq_h = d1[sq_d]
            img2 = img.resize(d1[sq_d])
            #img2.save(os.path.join(dir,"%s_%d.jpg"%(img_id,sq_d)))
            sq[sq_d]={}
            sq[sq_d][1] = img2.crop((0,0,sq_w,sq_w)) #top
            sq[sq_d][2] = img2.crop((0,int((sq_h/2.0)-(sq_w/2.0)),sq_w,int((sq_h/2.0)+(sq_w/2.0)))) #center
            sq[sq_d][3] = img2.crop((0,sq_h-sq_w,sq_w,sq_h)) #bottom
    else:
        for sq_d in d1:
            sq_w,sq_h = d1[sq_d]
            img2 = img.resize(d1[sq_d])
            #img2.save(os.path.join(dir,"%s_%d.jpg"%(img_id,sq_d)))
            sq[sq_d]={}
            sq[sq_d][1] = img2.crop((0,0,sq_h,sq_h)) #left
            sq[sq_d][2] = img2.crop((int((sq_w/2.0)-(sq_h/2.0)),0,int((sq_w/2.0)+(sq_h/2.0)),sq_h)) #center
            sq[sq_d][3] = img2.crop((sq_w-sq_h,0,sq_w,sq_h)) #right
    # (left, upper, right, lower)-tuple.
    for sq_d in d1:
        for sq_n in sq[sq_d]:
            img3 = sq[sq_d][sq_n]
            #img3.save(os.path.join(dir,"%s_%d_%d.jpg"%(img_id,sq_d,sq_n)))
            corner1 = img3.crop((0,0,299,299)) #top-left
            corner1.save(os.path.join(dir,"%s_%d_%d_%d.jpg"%(img_id,sq_d,sq_n,1)))
            corner2 = img3.crop((sq_d-299,0,sq_d,299)) #top-right
            corner2.save(os.path.join(dir,"%s_%d_%d_%d.jpg"%(img_id,sq_d,sq_n,2)))
            corner3 = img3.crop((0,sq_d-299,299,sq_d)) #bottom-left
            corner3.save(os.path.join(dir,"%s_%d_%d_%d.jpg"%(img_id,sq_d,sq_n,3)))
            corner4 = img3.crop((sq_d-299,sq_d-299,sq_d,sq_d)) #bottom-right
            corner4.save(os.path.join(dir,"%s_%d_%d_%d.jpg"%(img_id,sq_d,sq_n,4)))
            d2_sq = sq_d/2.0
            d2_299 = 299/2.0
            corner5 = img3.crop((int(d2_sq-d2_299),int(d2_sq-d2_299),int(d2_sq+d2_299),int(d2_sq+d2_299))) #center
            corner5.save(os.path.join(dir,"%s_%d_%d_%d.jpg"%(img_id,sq_d,sq_n,5)))
            
input_dir = "/home/users/nus/e0146089/t/transferred_train/"
output_dir = "/home/users/nus/e0146089/t/transferred_train_sorted/"
f = open("/home/users/nus/e0146089/train.csv","r")
skip=1
count = 0
for l in f.readlines():
   if skip>0:
      skip=skip-1
      continue
   (img_file,category) = l.strip().split(",")
   img_file=img_file.strip()
   img_id=img_file.split(".jpg")[0]
   cat_output_dir = os.path.join(output_dir,category)
   if os.path.exists(cat_output_dir)==False:
      os.makedirs(cat_output_dir)
   input_file = os.path.join(input_dir,img_file)
   preprocess(input_file,img_id,category,output_dir)
   print("%s -> %s"%(img_file,category))
f.close()