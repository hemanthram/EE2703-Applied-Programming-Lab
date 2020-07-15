import sys
import numpy as np
import math

def parse_val(x):
    y = len(x)
    if(not x[y-1].isalpha()):
        return float(x)
    if(x[y-1]=='p'):
        return float(x[0:y-1])* 1e-12   
    if(x[y-1]=='n'):
        return float(x[0:y-1])* 1e-9
    if(x[y-1]=='u'):
        return float(x[0:y-1])* 1e-6
    if(x[y-1]=='m'):
        return float(x[0:y-1])* 1e-3
    if(x[y-1]=='k'):
        return float(x[0:y-1])* 1e3
    if(x[y-1]=='M'):
        return float(x[0:y-1])* 1e6
    if(x[y-1]=='G'):
        return float(x[0:y-1])* 1e9

class comp:
	name = ''
	nodes = []
	value = 0
	global node,w
	def __init__(self, info):
		l = len(info)
		self.name = info[0]
		self.nodes = info[1:]
		self.nodes.pop()
		if(l == 4):
			if(self.name[0] == 'R' or self.name[0] == 'V' or self.name[0] == 'I'):
				self.value = parse_val(info[-1])
			elif(self.name[0] == 'L'):
				if(AC == 1):
					self.value = complex(0,parse_val(info[-1])*w)
				else:
					self.value = math.inf
			elif(self.name[0] == 'C'):
				if(AC == 1):
					self.value = complex(0,-1/(parse_val(info[-1])*w))
				else:
					self.value = 0
		elif(l == 5):
			self.nodes.pop()
			self.value = parse_val(info[-1])
		else:
			_ = [self.nodes.pop() for i in range(2)]
			phi = parse_val(info[-1])
			v = (parse_val(info[-2]))/2
			self.value = v*complex(math.cos(phi),math.sin(phi))
		for n in self.nodes:
			node[n] = True

# if(len(sys.argv) < 2):
# 	print('Enter file name')
# 	exit()
# filename = sys.argv[1]
filename = 'ckt.netlist'
try:
	f = open(filename, 'r')
except Exception:
	print('File not found')
	exit()
netlist = list(map(lambda x : x.strip('\n'),f.readlines()))
size = len(netlist)
start = 0
end = 0
AC = 0
w = 0
for i in range(size):
	if(netlist[i] == '.circuit'):
		start = i
	if(netlist[i] == '.end'):
		end = i
	if(netlist[i].split(' ')[0] == '.ac'):
		AC = 1
		w = parse_val(netlist[i].split(' ')[2])
		w = w*(math.pi)*2
		break
if(start >= end):
	print('Invalid .netlist file')
	exit()
netlist = netlist[start+1:end]
netlist = list(map(lambda x : (x.split('#')[0].strip(' ')).split(' '), netlist))

node = {}
Vs = {}
vs = 0 
ckt = [comp(x) for x in netlist]
n = 0
for k in node:
	node[k] = n; n+=1

for ele in ckt:
	if(ele.name[0] == 'V'):
		Vs[ele.name] = vs
		vs += 1

if(AC != 1):
	M = np.zeros((n+vs,n+vs))
	b = np.zeros(n+vs)
else:
	M = np.zeros((n+vs,n+vs),dtype = np.complex)
	b = np.zeros(n+vs,dtype = np.complex)

for ele in ckt:
	if(ele.name[0] == 'V'):
		high = node[ele.nodes[0]]
		low = node[ele.nodes[1]]
		M[n+Vs[ele.name]][high] = 1
		M[n+Vs[ele.name]][low] = -1
		b[n+Vs[ele.name]] = ele.value
		if(high != 0):
			M[high][n+Vs[ele.name]] = 1
		if(low != 0):
			M[low][n+Vs[ele.name]] = -1
	elif(ele.name[0] == 'R' or ele.name[0] == 'L' or ele.name[0] == 'C'):
		n1 = node[ele.nodes[0]]
		n2 = node[ele.nodes[1]]
		if(n1 != 0):
			M[n1][n1] = M[n1][n1] + 1/(ele.value)
			M[n1][n2] = M[n1][n2] - 1/(ele.value)
		if(n2 != 0):
			M[n2][n1] = M[n2][n1] - 1/(ele.value)
			M[n2][n2] = M[n2][n2] + 1/(ele.value)
	elif(ele.name[0] == 'I'):
		fr = node[ele.nodes[0]]
		to = node[ele.nodes[1]]
		b[fr] = b[fr] + ele.value
		b[to] = b[to] - ele.value
M[0][0] = 1
print(node)
print(M)
print(b)
x = np.linalg.solve(M,b)
print(x)