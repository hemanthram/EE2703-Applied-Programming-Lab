"""
EE2703 Assignment - 1
Reading .netlist file and printing details in reverse order

- Hemanth Ram (EE18B132)

"""


# importing sys library for getting input from command line
import sys

# If no file name entered, exiting 
if(len(sys.argv) < 2):
	print('Enter file name')
	exit()

# Reading file name
filename = sys.argv[1]

# Trying to open file, else, exiting
try:
	f = open(filename, 'r')
except Exception:
	print('File not found')
	exit()

# Reading lines of file and striping '\n' from each line
netlist = list(map(lambda x : x.strip('\n'),f.readlines()))

# Finding line number where '.circuit' and '.end' occurs and 
# storing in start and end respectively
size = len(netlist)
start = 0
end = 0
for i in range(size):
	if(netlist[i] == '.circuit'):
		start = i
	if(netlist[i] == '.end'):
		end = i

# Exiting if the netlist file is invalid
if(start >= end):
	print('Invalid .netlist file')
	exit()

# Extracting only the useful part of file (from .circuit to .end)
netlist = netlist[start+1:end]

# Checking for comments and removing them
l = len(netlist)		# Finding length of netlist
ntl = []		# ntl is the final list of lists which contains the details of elements 
for i in range(l):
	line = netlist[i]
	ind = line.find('#')	# Finding where "#" occurs or where comment starts
	if(ind == -1):			# If no comment present:
		ntl.append(line.split(' '))		# Appending line without change
		continue
	elif(ind == 0):			# If entire line is a comment, skipping it
		continue
	ntl.append(line[:ind].strip(' ').split(' '))	# If '#' present, considering only what is present before that

# Printing details of each element in reverse order as asked
for l in ntl:
	l.reverse()		# Reversing details of each element
	for ele in l:
		print(ele, end=' ')		# Printing the detail
	print('')