#!/usr/bin/python3

import sys

print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))

args = sys.argv[1].split(',')
print(args)

xmin = int(args[1])
ymin = int(args[2])
width = int(args[3])
height = int(args[4])

xmax = xmin + width
ymax = xmax + height

output = [str(xmin), str(ymin), str(xmax), str(ymax), str(width), str(height)]

print(','.join(output))