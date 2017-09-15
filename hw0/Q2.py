import sys
import math
from PIL import Image

infile=Image.open(sys.argv[1]);
pic=list(infile.getdata());
width,length=infile.size;

new_pixel=[0,0,0,0];
new_pic=[];

for pixel in pic:
	i=0;
	for rgb in pixel:
		new_pixel[i]=int(math.floor(rgb/2));
		i+=1;
	new_pic.append((new_pixel[0],new_pixel[1],new_pixel[2],new_pixel[3]));

outfile=Image.new(infile.mode,infile.size);
outfile.putdata(new_pic);
outfile.save("Q2.png");
