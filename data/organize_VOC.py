print(os.path.exists('../data/train.txt'))
f = open('../data/train.txt', 'r')
lines = f.readlines()

for line in lines:
    #print(line.split('/')[-1][:-1])
    line = "/".join(line.split('/')[2:])
    
    if (os.path.exists(line[:-1])):
        os.system("cp "+ line[:-1] + " VOC/images/train")
        
print(os.path.exists('../data/train.txt'))
f = open('../data/train.txt', 'r')
lines = f.readlines()

for line in lines:
    #print(line.split('/')[-1][:-1])
    line = "/".join(line.split('/')[2:])
    line = line.replace('JPEGImages', 'labels')
    line = line.replace('jpg', 'txt')
    #print(line)
    if (os.path.exists(line[:-1])):
        os.system("cp "+ line[:-1] + " VOC/labels/train")

print(os.path.exists('../data/2007_test.txt'))
f = open('../data/2007_test.txt', 'r')
lines = f.readlines()

for line in lines:
    #print(line.split('/')[-1][:-1])
    line = "/".join(line.split('/')[2:])
    
    if (os.path.exists(line[:-1])):
        os.system("cp "+ line[:-1] + " VOC/images/val")

print(os.path.exists('../data/2007_test.txt'))
f = open('../data/2007_test.txt', 'r')
lines = f.readlines()

for line in lines:
    #print(line.split('/')[-1][:-1])
    line = "/".join(line.split('/')[2:])
    line = line.replace('JPEGImages', 'labels')
    line = line.replace('jpg', 'txt')
    #print(line)
    if (os.path.exists(line[:-1])):
        os.system("cp "+ line[:-1] + " VOC/labels/val")