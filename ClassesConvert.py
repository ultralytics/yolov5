

def ClassesConvert():
    f = open("./classes.txt",'r')
    w = open("classes_list.txt", 'w')
    classes = { int(line[:line.find(' ')].strip()):line[line.find(' ')+1:].strip() for line in f.read().split("\n")[:-1]}

    for c,v in classes.items():
        print("{:04d} -  {:s}\n".format(c, v))

    l= []
    maxkey = max(classes)
    for i in range(max(classes)+1):
        if i in classes.keys():
            l.append(classes[i])
        else:
            l.append('Non ID Obj')
        print("{:04d} -  {:s}\n".format(i, l[i]))
        print("-"+ l[i], file=w)
    w.close()

if __name__ == "__main__":
    ClassesConvert()
