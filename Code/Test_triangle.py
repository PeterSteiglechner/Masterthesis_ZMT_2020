import numpy as np


triangle_points = (np.array([0,0]), np.array([0,1]), np.array([1,1])) 

import scipy.optimize

def q(a,b):
	# this gives the point on the line at which I mirror, given points a,b
	v = (b-a)/np.sqrt(np.dot((b-a),(b-a)))
	return lambda s: a+s*v
def scalarprod(s,a,b,p):
	Q = q(a,b)(s)
	# Note I minimimize this function below. Why 2 terms? Because if only first term, there's a minimum if Q==a. Which will be chosen
	return np.dot(Q-a,Q-p) + np.dot(Q-b,Q-p) 

def project_triangle(p):
	global triangle_points
	a,b,c = triangle_points
	#solve via barycentric coordinates
	def barycentric(x,a,b,c,p):
		return p - (a + (b - a) * x[0] + ( c- a) * x[1])
	x0 = np.array([0.01, 0.2])
	x = scipy.optimize.root(barycentric, x0, args=(a,b,c,p,))
	s = x.x[0]
	t = x.x[1]
	print("Barycentric coordinates: ", '%.2f' % s, ", ", '%.2f' % t)
	print("    Point retrieved is: ", a + (b - a) * s + ( c- a) * t , " with p-bary = ", barycentric([s,t],a,b,c,p))
	if (s>=0) and (s<=1) and (t>=0) and (t<=1) and (s+t<=1):
		print("In triangle")
		return p
	else:
		print("Projected")
		# Find mirror points 
		mirrorpts=[]
		for n,(p1,p2) in enumerate([(a,b), (a,c), (b,c)]):
			vector = (p2-p1)/np.sqrt(np.dot((p2-p1), (p2-p1)))
			print("Vector: ", vector)
			s0=0.001
			s = scipy.optimize.root(scalarprod, s0, args=(p1,p2,p))
			Q = q(p1,p2)(s.x)
			print("Optimal s: ", s.x, " giving Q =", Q)
			#p_mirror = p+2*(Q-p)		
			mirrorpts.append(Q)
		print(mirrorpts)
		distances = np.array([np.dot(qc- p, qc-p) for qc in mirrorpts])
		minimum = np.min(distances)
		print(distances)
		projected_point = mirrorpts[np.where(distances==minimum)[0][0]]
		print("Proje Point", projected_point)
		return projected_point
	return
	
a,b,c = triangle_points
print("Triangle: ",a,b,c)

p=np.array([0.2, 0.6])
project_triangle(p)

p=np.array([1, 1])
project_triangle(p)

p=np.array([0.4, 1.4])
project_triangle(p)


p=np.array([1, 0])
project_triangle(p)

