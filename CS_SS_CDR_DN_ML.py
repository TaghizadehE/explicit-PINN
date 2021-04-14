'''
Central space steady state convection diffusion Monod reaction left dirichlet right neumann finite difference
0 = - Deff * u_xx + V * u_x - R * u/(u+Ka) + C
    
with Dirichlet boundary conditions at x=0: u(x0,t) = 0, 
with Neumann boundary conditions at x=L: u_x = 0
and initial condition u(x,0) = 

The discrete solution is unreliable when dx > 2*D/[V*(b-a)]
'''
import time
import numpy as np
import pandas as pd
from scipy import sparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
import tensorflow.keras.models
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy import interpolate
# from deep_regression_v6 import Deep_Regression
from Auxiliary import relz


# function for normalazing the data
def normalize_data_in(X):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X)
    normalized_X = scaler.transform(X)
    return normalized_X

# function to compute the effective diffusion
def D_star(Pe_):
    data = pd.read_csv('data/effective_dispersion_Pe_u_brain_liver__allrange_correct_T.csv', header=4).to_numpy()
    Pe = data[:,1]
    D_star = data[:,2]
    ius = InterpolatedUnivariateSpline(Pe, D_star)
    return ius(Pe_)

def min_max_training_set_csvinput(input_file_name, nb_realization):
    data = pd.read_csv(input_file_name, header=0).to_numpy()
    data = data[:,:-1]
    mins = pd.DataFrame(data).min().to_numpy().reshape(1,-1)
    maxs = pd.DataFrame(data).max().to_numpy().reshape(1,-1)
    return mins, maxs


# set parameters to load from saved learned model

#@
nb_realization = relz(64)
nb_epoch = 2000

# different cases on how model learned (default: case3)
case1  = '__Phi2_I__'
case2  = '__Phi2_II__'
case3  = '__Phi2_III__'
case32 = '__Phi2_III__Pe_II__'

#@
case_name = case3

common_name = case_name+'R'+str(nb_realization)+'_E'+str(nb_epoch) 

input_file_name = 'data/learning_parameters_6features_1target_Pe_u_Brinkman'+case_name+'R'+str(nb_realization)+'.csv'
save_file_name  = 'models__Pe_u/mlp_model'+common_name #+'_final'
print('load learned model from:', save_file_name)

load_model = tensorflow.keras.models.load_model(save_file_name)


# select simluation case: brian or liver

def simul_case_(argument):
    switcher = {
        1: 'base',
        2: 'brain',
        3: 'liver',
    }
    return switcher.get(argument, "nothing")

#@
simul_case    = simul_case_(3)  

# Pe brain=8,  Pe liver=52
if simul_case == 'brain':
    simul_case_Pe = 8
elif simul_case == 'liver':
    simul_case_Pe = 52
else:
    print('sth is wrong in Pe!')


print('solving:', simul_case)


    
if simul_case == 'brain':
    xL     = 4*5e-6    # [m]
    rr     = 0.47e-6    # [m] effective radius
    epsl_B = 0.260372
    if simul_case_Pe == 1:
        Dr = 0.1
        Ka = 10
        km = 190.1312811
        cL = 2.0
        Pe = 1.0196650934
    elif simul_case_Pe == 8:
        Dr = 0.1
        Ka = 1
        km = 226.34676324128571
        cL = 2.0
        Pe = 8.15732074795076
    elif simul_case_Pe == 10:
        Dr = 0.1
        Ka = 1
        km = 316.8854685378
        cL = 2.0
        Pe = 10.012252848597623
    else:
        print('sth is wrong!')
    
elif simul_case == 'liver':
    xL     = 4*170e-6    # [m]
    rr     = 11.67e-6    # [m] effective radius
    epsl_B = 0.18724390    
    if simul_case_Pe == 48:
        Dr = 0.19191 
        Ka = 1
        km = 5.874193858
        cL = 22.1404
        Pe = 48.0840777
    elif simul_case_Pe == 52:
        Dr = 0.1
        Ka = 1.0
        km = 1.5
        cL = 2.0
        Pe = 52.2210168759
    else:
        print('sth is wrong!')

epsl_s = 1 - epsl_B    
# Pe=r*V/Df
Df = 1e-10
C = 0
phi2 = km*rr**2/(Ka*Df)
Deff = D_star(Pe)
print('Deff:', Deff)
V = Pe*Df/rr

print('phi2:',phi2,' Dr:',Dr,' Pe:',Pe)

#@
M = 1001    # GRID POINTS on space interval
N = 20000   # number of iteration in Relaxation Method for Nonlinear Finite Differences

x0 = 0      # [m]

# ----- Spatial discretization step -----
dx = (xL - x0)/(M - 1)


stable = 2*Deff/(V*(xL-x0))
print('check if grid discretization is stable: dx=',dx,'----- < (?)----','2*D/[V*(b-a)]=', stable)
assert dx < stable

p1 = Deff/dx**2
p2 = V/(2*dx)
p = -Deff/dx**2 - V/(2*dx)
q = -Deff/dx**2 + V/(2*dx)


xspan  = np.linspace(x0, xL, M)

# ----- Initializes matrix U -----
U = np.zeros((M))

# ----- Initializes matrix dU/dx -----
Ux = np.zeros((M))

#----- Initial condition -----
U = np.ones((M))*cL
# linear line from 22.14 to 0 for initial condition
# U = np.linspace(xL, x0, M)

#----- Dirichlet boundary conditions at x=0 -----
U[0] = cL

#----- Neumann boundary conditions a x=L -----
# g = np.ones((N))*5.0
g = 0.0

# ----- Initializes matrix R (reaction rate) -----
# R = np.ones((M))*10000.0
# instead of (72) --> learning algor
# eta_DNS = -39044*xspan + 1.0869
# make a function for eta
# R_DNS = epsl_s/epsl_B*km*eta_DNS

#@
tol = 5e-10
norms = []

# porosity = epsl_B
epsl_B_ = np.repeat(epsl_B,M).reshape(-1,1)
phi2_   = np.repeat(phi2,M).reshape(-1,1)
Pe_     = np.repeat(Pe,M).reshape(-1,1)
Dr_     = np.repeat(Dr,M).reshape(-1,1)


min_dataset, max_dataset = min_max_training_set_csvinput(input_file_name, nb_realization)

n_ML = 0

start = time.time()

for j in range(N):
    # compute the derivate of c (dc/dx), x=0 forward, x=L backward, center for the rest
    Ux[0] = (U[1]-U[0])/dx
    Ux[1:M-1] = (U[2:M]-U[0:M-2])/(2*dx)
    Ux[M-1] = (U[M-1]-U[M-2])/dx
    grad = np.gradient(U, dx)
    assert Ux.all() == grad.all()
    # multiply by 'r' for all d2-d5
    Ux_normalized = -1*rr*Ux
    U_normalized  = U/Ka
    
    # call ML computation every 5 iteration
    if j % 10 == 0:
#         if Ux.min() < min_dataset[0,4]:
#             continue
        n_ML += 1
        # compute eta from ML
        # feature sequence: (1) phi2, (2)Pe, (3)Dr, (4)c_f_avg_norm, (5)d_f_c_avg_norm,  
        # target: (6) eta_d_f
        features = np.concatenate((epsl_B_, phi2_, Pe_, Dr_, U_normalized.reshape(-1,1), Ux_normalized.reshape(-1,1)), axis=1)
        combine  = np.concatenate((min_dataset, max_dataset, features), axis=0)
        normalize_features_combine = normalize_data_in(combine)
        normalize_features = normalize_features_combine[2:]
        eta_predicted = load_model.predict(normalize_features).flatten()
    
    
    R = epsl_s/epsl_B*km*eta_predicted    

    Uold = U.copy()
    r = 2*Deff/dx**2 + R[1:M-1]/(U[1:M-1]+Ka)
    U[1:M-1] = -U[0:M-2]*(p/r) - U[2:M]*(q/r) - C/r
    rlast = 2*Deff/dx**2 + R[M-1]/(U[M-1]+Ka)
    U[M-1] = -U[M-2]*(p/rlast) - (U[M-2]+2*dx*g)*(q/rlast) - C/rlast
    norm = np.linalg.norm(U-Uold)
    norms.append(norm)
    if norm < tol:
        print('norm is less than ', tol, 'after', j, 'iteration')
        # plt.plot(norms
        break

end = time.time()
print('Time:', end-start)
if norm < tol:
    print('norm did converge; it is: ', norm, 'after',j, 'iteration')
else:
    print('norm did NOT converge; it is: ', norm, 'after',j, 'iteration')
    
        

print('number of times ML used:', n_ML)
# ----- Checks if the solution is correct:
# check = np.allclose(np.dot(A,U[1:M]), np.add(b,-K))
# print(check)




if simul_case == 'brain':
    c_2D_mb = pd.read_csv('data/brain_average_c_moving_bar_2D_Brinkman_realdata_Pe_'   +str(simul_case_Pe)+'.csv', header=None).to_numpy()
    c_2D_e  = pd.read_csv('data/brain_average_c_exact_average_2D_Brinkman_realdata_Pe_'+str(simul_case_Pe)+'.csv', header=None).to_numpy()
     
elif simul_case == 'liver':
    c_2D_mb = pd.read_csv('data/liver_average_c_moving_bar_2D_Brinkman_realdata_Pe_'   +str(simul_case_Pe)+'.csv', header=None).to_numpy()
    c_2D_e  = pd.read_csv('data/liver_average_c_exact_average_2D_Brinkman_realdata_Pe_'+str(simul_case_Pe)+'.csv', header=None).to_numpy()


# c_2D = c_2D_e

fig1 = plt.figure()
plt.plot(xspan, U, 'r', label='ML')
plt.plot(c_2D_e[:,0]*1e-6,  c_2D_e[:,1],  'b', label='moving bar 2D DNS exact')
plt.plot(c_2D_mb[:,0]*1e-6, c_2D_mb[:,1], 'g', label='moving bar 2D DNS moving bar')
plt.title(simul_case)
plt.yticks(np.arange(0.0, 2.01, step=0.2))
plt.grid(True)
plt.legend()


fig2 = plt.figure()
plt.plot(xspan, eta_predicted, 'k', label='Machine learned $\eta$')
plt.yticks(np.arange(0.0, 1.01, step=0.2))
plt.title(simul_case)
plt.grid(True)
plt.legend()


# compute error
r_eff_vornoi_brain = 0.47e-6   # m
r_eff_vornoi_liver = 11.67e-6  # m 

if simul_case == 'brain':
    r_mb = r_eff_vornoi_brain/3
elif simul_case == 'liver':
    r_mb = r_eff_vornoi_liver/3

# find the position of the 
for index, el in enumerate(xspan):
    if el >= r_mb:
#         print(el, r_mb) 
        print('location of cutting the data:', index)
        break
cut = index

c_2D_mb_interpolate = InterpolatedUnivariateSpline(c_2D_mb[:,0]*1e-6, c_2D_mb[:,1])
# c_2D_mb_interpolate_v2 = interpolate.interp1d(c_2D_mb[:,0]*1e-6, c_2D_mb[:,1])
error_U_mb   = np.divide(abs(U[cut:]-c_2D_mb_interpolate(xspan)[cut:]), U[cut:])*100
error_ave_mb = sum(error_U_mb)/len(error_U_mb)
print('error of moving average:', error_ave_mb)



c_2D_e_interpolate = InterpolatedUnivariateSpline(c_2D_e[:,0]*1e-6, c_2D_e[:,1])
error_U_e   = np.divide(abs(U-c_2D_e_interpolate(xspan)), U)*100
error_ave_e = sum(error_U_e)/len(error_U_e)
print('error of exact:', error_ave_e)


fs1  = 18

if simul_case == 'brain':
    label1 = ''
    label2 = ''
    xrange = np.arange(0, 20.01, step=2.5)
    xleftletter = -5.5
    letter1 = 'B'
    letter2 = 'C'
    figsavename = 'figs/brain_c_eta.pdf'
    figsavename_smooth = 'figs/brain_c_eta_smooth.pdf'
    
elif simul_case == 'liver':
    label1 = 'Macroscale + ML'
    label2 = 'Microscale Average'
    xrange = np.arange(0, 700.01, step=100)
    xleftletter = -5.5*700/20
    letter1 = 'E'
    letter2 = 'F'
    figsavename = 'figs/liver_c_eta.pdf'
    figsavename_smooth = 'figs/liver_c_eta_smooth.pdf'


fig1, ax = plt.subplots(figsize=(7,6*1.0), nrows=2, ncols=1, dpi=200)
plt.subplots_adjust(hspace=0.2)



# img = mpimg.imread('../figs/liver_Pe_52_c.png')
# ax[0].imshow(img, aspect='auto')
# ax[0].axis('off')
# ax[0].tick_params(labelbottom=False, labelleft=False) 

ax[0].plot(xspan*1e6, U,                       'r', label=label1, linewidth=2)
# ax[0].plot(c_2D_e[:,0]*1,     c_2D_e[:,1 ],    'b', label=label2, linewidth=2)
ax[0].plot(c_2D_mb[cut:,0]*1, c_2D_mb[cut:,1], 'g', label=label2, linewidth=2)

# plt.title('R ML')
# ax[0].set_xlabel('L ($\mu m$)', fontsize=fs1)
ax[0].tick_params(labelbottom=False, labelsize=fs1-6)    
ax[0].set_ylabel(r'$\langle c_\beta \rangle^\beta ~(\frac{mol}{m^3}$)', fontsize=fs1)
ax[0].set_xticks(xrange)
# ax[0].tick_params(direction='out', length=6, width=2, colors='r', grid_color='r', grid_alpha=0.5)
#ax1.set_ylabel(self.labely, fontsize=fs1)
# plt.xticks(np.arange(0, xL, step=xL/4))
# plt.yticks(np.arange(0, 22.1, step=2))
ax[0].set_ylim([0,2.1])
ax[0].grid(True)
ax[0].legend(fontsize=fs1-4)
ax[0].text(xleftletter, 0.90, letter1, fontsize=fs1+4)


ax[1].plot(xspan*1e6, eta_predicted, 'k', linewidth=2)
ax[1].set_xlabel('$L ~(\mu m$)', fontsize=fs1)
ax[1].set_ylabel('Machine learned $\eta$', fontsize=fs1)
ax[1].set_ylim([0,1])
ax[1].set_xticks(xrange)
ax[1].tick_params(labelsize=fs1-6) 
plt.grid(True)
ax[1].text(xleftletter, 0.45, letter2, fontsize=fs1+4)


fig1.tight_layout()
# plt.savefig(figsavename, dpi=300)
# plt.savefig(figsavename_smooth, dpi=300)
