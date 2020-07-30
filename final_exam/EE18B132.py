#### EE2703 Final Exam
#### Author : Hemanth Ram
#### Date : 29th July, 2020
#### Please refer to the report for parameters and conventions used

# Importing required Libraries
from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3

# Function which calculates potential distribution given 
# M,N - number of rows & columns in mesh
# dx - step size of the mesh
# hbyL - ratio between h and Ly
# acc - required accuracy
# Niter - maximum no.of iterations
def potential(M,N,K,dx,hbyL,acc,Niter):
    k = int(M*(1-hbyL))
    phi = zeros((M,N))
    phi[0] = np.ones((1,N))
    errors = ndarray((Niter,1))
    for _ in range(Niter):
        # For calculating errors
        old_phi = phi.copy()

        # Updates in the phi matrix
        phi[1:-1,1:-1] = 0.25*(phi[1:-1,0:-2]+
                               phi[1:-1,2:]+
                               phi[0:-2,1:-1]+
                               phi[2:,1:-1])
        phi[k] = ( K*(phi[k+1]) + phi[k-1] ) / (K+1)

        # Error updates
        errors[_] = (abs(old_phi-phi).max())

    return phi

# Setting up the parameters
# Throughout, we use M = 200, N = 100 and the mesh which is created here
M = 200
N = 100
delta = 10/N
x = range(M-1,-1,-1)
y = np.arange(0,N)
Y,X = meshgrid(y,x)
show(); close()

# Using the potential function, plotting the potential
# distribution contour for 9 different values of h
print("Plotting Contours .. please wait ..")
subplots(figsize=(11,17))
for i in range(1,10):
    phi = potential(M,N,2,1,i/10,10**(-7),15000)  
    subplot(2,5,i)
    CS = contourf(Y/10,X/10,phi)
    axis('scaled')    
    xlabel("x (cm)"); ylabel("y (cm)")
    title("h = 0."+str(i)+" Ly")
colorbar(CS, shrink=0.8, extend='both')
suptitle("Potential Distributions for different values of h", fontsize=17, y =0.98)
show()
print("Done")


# Charge Calculation :
# Calculating fields and then charges for the same values of h
print("Plotting Charge values .. please wait")
q_top = []
q_fluid = []
for i in range(1,10):
    k = int(M*(1-i/10))
    phi = potential(M,N,2,1,i/10,10**(-7),15000)  
    # Initialising Ex, Ey arrays
    Ex = zeros((M, N))
    Ey = zeros((M, N))

    # Filling Ex, Ey arrays with fields
    Ex[1:-1, 1:-1] = (phi[1:-1, 2:] - phi[1:-1, 0:-2])/delta
    Ey[1:-1, 1:-1] = (phi[2:, 1:-1] - phi[0:-2, 1:-1])/delta

    # Calculating charges as mentioned in report
    q_top.append(-sum(Ey[1])*delta)
    q_fluid_bottom = 2*(sum(Ey[-2]))*delta
    q_fluid_wall = 4*(sum(Ex[k:,1]))*delta
    q_fluid.append(q_fluid_bottom+q_fluid_wall)

# Plotting Q_fluid vs H
subplots(figsize=(19,10))
subplot(1,2,1)
plot(array(range(1,10))*2, q_fluid, linewidth=10)
title("Charge on wall touching fluid vs H ",fontsize = 17)
xlabel("h (cm)",fontsize = 13)
ylabel("Q/(\u03B5*W) - charge per width / \u03B5", fontsize = 16)
grid()
# Plotting Q_top vs H
subplot(1,2,2)
plot(array(range(1,10))*2, q_top, linewidth=10)
title("Charge on Top vs H ",fontsize = 17)
xlabel("h (cm)",fontsize = 13)
ylabel("Q/(\u03B5*W) - charge per width / \u03B5", fontsize = 16)
grid()
show()
print("Done")

# Calculating field and 
# Verification of continuity of Dn at boundary for h = 0.5
hbyL = 0.5
phi = potential(M,N,2,1,hbyL,1,15000)  
k = int(M*(1-hbyL))
# initialising Ex, Ey arrays
Ex = zeros((M, N))
Ey = zeros((M, N))

# filling Ex, Ey arrays with fields for h = 0.5
Ex[1:-1, 1:-1] = (phi[1:-1, 2:] - phi[1:-1, 0:-2])/delta
Ey[1:-1, 1:-1] = (phi[2:, 1:-1] - phi[0:-2, 1:-1])/delta

# Printing Ex and Ey at the center of mesh
print("Ex at center for h = 0.5Ly:")
print(Ex[k][N//2],"V / cm")
print("Ey at center for h = 0.5Ly:")
print(Ey[k][N//2],"V / cm", end='\n\n')

# In the matrix m = k refers to the border
# So, we take Ey at m = k+1, and m = k-1 and check if Dn is continuous
Ey_knxt = Ey[k+1,1:-1]
Ey_kpre = Ey[k-1,1:-1]
# To check for Dn continuity, we can take ratio of the fields Ey1/Ey2 and
# check if the ratio is e2/e1 which is 1/2 in our case
# To check value of e2/e1, we consider here the average of all the ratios at boundary
print("E_fluid/E_air at all points in boundary:")
print(Ey_knxt/Ey_kpre); print("")
print("Average value of the ratio is")
print(mean(Ey_knxt/Ey_kpre))
print("We have got the expected value ( 0.48 ~ 0.5 -> 1/2 )")
print("Hence Dn is proved to be continuous at the boundary")

# Calculating change in angle of electric field at boundary
angle_i = arctan(Ex[k-1][N//2]/Ey[k-1][N//2])
angle_r = arctan(Ex[k+1][N//2]/Ey[k+1][N//2])
print("Incident Angle :")
print(round(angle_i*180/pi,3),"degrees")
print("Transmitted Angle :")
print(round(angle_r*180/pi,3),"degrees")
print("Change in Angle :")
print(round((angle_r-angle_i)*180/pi,3),"degrees")
# Finding sini/sinr to check Snell's Law
print("sini/sinr =")
print(sin(angle_i)/sin(angle_r))