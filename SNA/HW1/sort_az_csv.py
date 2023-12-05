#py script to sort csv line by line alphabetically
# Usage: python sort_az_csv.py <input_file> <output_file> or just python sort_az_csv.py <input_file> to overwrite input file
# Example: python sort_az_csv.py input.csv output.csv

import sys
import csv

#read in file
with open(sys.argv[1], 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)

#sort list
your_list.sort()
#remove empty lines
your_list = [x for x in your_list if x != []]
print(your_list[:10])

#write out file in same directory
if len(sys.argv) < 3:
    with open(sys.argv[1], 'w') as f:
        writer = csv.writer(f)
        writer.writerows(your_list)
else:
    with open(sys.argv[2], 'w') as f:
        writer = csv.writer(f)
        writer.writerows(your_list)