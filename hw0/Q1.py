import sys
infile=open(sys.argv[1],'r');
outfile=open('Q1.txt','w');
text=infile.read();

words=[];
count=[];
temp='';

for letter in text:
	if letter!=' ' and letter!='\n':
		temp+=letter;
	else:
		i=0;
		j=0;
		for word in words:
			if temp==word:
				count[i]+=1;
				j+=1;
				break;
			i+=1;
		
		if j==0:
			words.append(temp);
			count.append(1);
		temp='';

i=0;
for word in words:
	outfile.write(word),
	outfile.write(' '),
	outfile.write(str(i)),
	outfile.write(' '),
	outfile.write(str(count[i]));
	outfile.write('\n'),
	i+=1;
