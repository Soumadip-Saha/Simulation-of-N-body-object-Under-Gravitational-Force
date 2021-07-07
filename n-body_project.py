import time
start_time = time.time()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


## CONSTANT DEFINING SECTION ##

G = 6.67e-11
MAU = 1.0e30
AU = 1.5e11
VAU = 10000
daysec = 24.0*60*60
K1 = VAU*daysec/AU
K2 = G*daysec*MAU/(VAU*AU**2)
K3 = G*MAU**2/AU
K4 = MAU*VAU**2
K5 = MAU*VAU*AU
dt = 0.01

## FUNCTION DEFINING SECTION ##

# CALCULATES ANGULAR MOMENTUM OF BODY NUMBER n AT TIME = t #
def Angular_Momentum(R,V,M,n):
    r_n = R[n]
    v_n = V[n]
    M_n = M[n]
    L = M_n*np.cross(r_n,v_n)
    return K5*L

# CALCULATES POTENTIALL OF BODY NUMBER n AT TIME = t #
def Potential(R,M,n):
    r_bar =  np.delete(R,n,axis=0)
    r_n = R[n]
    r = r_bar - r_n
    M_bar = np.delete(M,n)
    M_n = M[n]
    modr = np.linalg.norm(r,axis = 1)
    temp = M_n*M_bar/modr
    pot = (-1)*np.sum(temp, axis = 0)
    return K3*pot

# CALCULATES KINETIC ENERGY OF BODY NUMBER n AT TIME = t #
def Kinetic(V,M,n):
    v_n = V[n]
    modv = np.linalg.norm(v_n)
    M_n = M[n]
    kin = 0.5*M_n*modv**2
    return K4*kin

# CALCULATES ACCELARATION OF BODY NUMBER n AT TIME = t #
def Accelaration(R,V,M,n):
    r_bar =  np.delete(R,n,axis=0)
    r_n = R[n]
    r = r_bar - r_n
    M_bar = np.delete(M,n)
    M_n = M[n]
    modr = np.linalg.norm(r,axis = 1)
    modr3byM = modr**3/M_bar
    temp = r/modr3byM[:,None]
    acc = np.sum(temp, axis = 0)
    return K2*acc

# FOR DERIVATIVE OF THE POSITION #
def f1(R,V):
    dXdt = K1*V
    return dXdt

# FOR DERIVATIVE OF THE ACCELARATION #
def f2(R,V,M):
    N = M.shape[0]
    dVdt = np.empty((N,3))
    for i in range(N):
        dVdt[i] = Accelaration(R,V,M,i)
    return dVdt

# 4TH ORDER RUNGE KUTTA METHOD #
def RK4(R,V,M):
    k11 = dt*f1(R,V)
    k21 = dt*f2(R,V,M)
    k12 = dt*f1(R+0.5*k11,V+0.5*k21)
    k22 = dt*f2(R+0.5*k11,V+0.5*k21,M)
    k13 = dt*f1(R+0.5*k12,V+0.5*k22)
    k23 = dt*f2(R+0.5*k12,V+0.5*k22,M)
    k14 = dt*f1(R+k13,V+k23)
    k24 = dt*f2(R+k13,V+k23,M)
    R += (k11+2*k12+2*k13+k14)/6;
    V += (k21+2*k22+2*k23+k24)/6;
    return R,V


## USER INPUT SECTION ##

N = int(input("Enter number of Bodies : "))
print("Enter all position in Astronomical units which is 1.5e11 m")
x = np.empty((N,3))
# MY INPUT #
## For Body 1 x0=0.0, y0=0.0, z0=0.0 ##
## For Body 2 x0=1.0, y0=0.0, z0=0.0 ##
## For Body 3 x0=0.0, y0=0.0, z0=1.0 ##
for i in range(N):
    x[i,0], x[i,1], x[i,2] = input("Enter initial position of body "+str(i+1)+" : ").split()
print("Enter all velocity in 10000 m/s units")
v = np.empty((N,3))
# MY INPUT #
## For Body 1 v_x0 = 0.0, v_y0 = 0.0, v_z0 = 0.0 ##
## For Body 2 v_x0 = 1.0, v_y0 = 2.0, v_z0 = 0.0 ##
## For Body 3 v_x0=-2.5, v_y0=-2.5, v_z0 = 0.0 ##
for i in range(N):
    v[i,0], v[i,1], v[i,2] = input("Enter initial velocity of body "+str(i+1)+" : ").split()
print("Enter all Masses in e30 kg units")
M = np.empty((N,))
for i in range(N):
    M[i] = input("Enter Mass of "+str(i+1)+" body : ")

## MAIN BODY ##



t = 0.0
# TO STORE VALUES FOR PLOTIING #
Nlist = []
Potlist = []
Kinlist = []
AMlist = []
# TO SET THE LIMITS OF AXIS IN PLOTTING #
max_x = 0
min_x = 0
max_y = 0
min_y = 0
max_z = 0
min_z = 0
# CREATING THE LIST PROPERLY TO STORE VALUES OF EACH BODY #
for i in range(N):
    Nlist.append([[],[],[]])
    AMlist.append([])
    Potlist.append([])
    Kinlist.append([])

# TIME COUNTING LOOP FOR THE PLOTS #
T = np.arange(0.0,1000.01,0.01)
# MAIN LOOP TO UPDATE POSITION FOR EACH BODY BY TIME dt #
while (t<1000.0):
    t += dt
    for i in range(N):
        # STORING THE MAX VALUE OF X,Y,Z POSITION IN STARS FOR PLOTIING IN A BETTER WAY #
        max_x = max(max_x,x[i,0])
        min_x = min(min_x,x[i,0])
        max_y = max(max_y,x[i,1])
        min_y = min(min_y,x[i,1])
        max_z = max(max_z,x[i,2])
        min_z = min(min_z,x[i,2])
        tempAM = Angular_Momentum(x,v,M,i)
        AMlist[i].append(tempAM)
        Nlist[i][0].append(x[i,0])
        Nlist[i][1].append(x[i,1])
        Nlist[i][2].append(x[i,2])
        AMlist[i][0]
        Potlist[i].append(Potential(x,M,i))
        Kinlist[i].append(Kinetic(v,M,i))
    x,v = RK4(x,v,M)
Nlist = np.array(Nlist)
AMlist = np.array(AMlist)
# SUMMED ALL THE ANGULAR MOMENTUM OF ALL THE BODIES FOR EACH TIME #
TotalAM = np.sum(AMlist,axis = 0)
Potlist = np.array(Potlist)
Kinlist = np.array(Kinlist)
# SUMMED ALL THE POTENTIALL OF ALL THE BODIES FOR EACH TIME #
TotalPot = np.sum(Potlist, axis = 0)
# SUMMED ALL THE KINETIC ENERGY OF ALL THE BODIES FOR EACH TIME #
TotalKin  = np.sum(Kinlist, axis = 0)
# TOTAL ENERGY OF THE SYSTEM FOR EVERY TIME #
TotalE = TotalKin+TotalPot

print("Done")
print("--- %s seconds ---" % (time.time() - start_time))
## PLOTTING ANIMATION SECTION ##

# TO PLOT ENERGY CONSERVATION FOR EACH TIME IN THE SYSTEM #
plt.plot(T,TotalE, label = "Total Energy")
plt.plot(T,TotalKin, label = "Total Kinetic Energy")
plt.plot(T,TotalPot,"--", label = "Total Potential Energy")
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Energy Profile')
plt.legend()
plt.savefig('Energy Profile.png')
plt.show()

# TO PLOT ANGULAR MOMENTUM CONSERVATION FOR EACH TIME IN THE SYSTEM #
plt.plot(T,TotalAM[:,0], label = "Total Angular Momentum in X direction")
plt.plot(T,TotalAM[:,1], label = "Total Angular Momentum in Y direction")
plt.plot(T,TotalAM[:,2], label = "Total Angular Momentum in Z direction")
plt.xlabel('Time')
plt.ylabel('Angular Momentum')
plt.title('Time vs Angular Momentum')
plt.legend()
plt.savefig('Angular Momentum Profile.png')
plt.show()

# TO PLOT A MOVING ANIMATION OF THE BODIES IN THEIR RESPECTIVE ORBIT #
fig=plt.figure(figsize=(15,15))
ax=fig.add_subplot(111,projection="3d")
line = [None]*N
scat = [None]*N
for i in range(N):
    line[i] = ax.plot([],[],[])[0]
    scat[i] = ax.plot([],[],[],marker = "o", label="Body "+str(i))[0]
ax.set_xlim(min_x-0.15, max_x+0.15)
ax.set_ylim(min_y-0.15, max_y+0.15)
ax.set_zlim(min_z-0.15, max_z+0.15)

def update(frame):
    global Nlist
    if (frame>5000):
        q = frame-5000
    else:
        q = 0
    for i in range(N):
        line[i].set_xdata(Nlist[i][0][q:frame])
        line[i].set_ydata(Nlist[i][1][q:frame])
        line[i].set_3d_properties(Nlist[i][2][q:frame])
        scat[i].set_xdata(Nlist[i][0][frame])
        scat[i].set_ydata(Nlist[i][1][frame])
        scat[i].set_3d_properties(Nlist[i][2][frame])
    graph = [item2 for item2 in scat]
    graph += [item1 for item1 in line]
    return graph


ax.set_xlabel("x-coordinate",fontsize=14)
ax.set_ylabel("y-coordinate",fontsize=14)
ax.set_zlabel("z-coordinate",fontsize=14)
ax.set_title("Visualization of orbits of stars in a N-body system\n",fontsize=14)
ani1 = animation.FuncAnimation(fig, update, frames=np.arange(0,np.shape(Nlist)[2],1000), blit=True)
ax.legend(loc="upper left",fontsize=14)
ani1.save('animation_final.mp4')
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))
