import numpy as np
import pandas as pd


def relz(argument):
    switcher = {
        1:  128,   # 128*1  = 128  = 2^7  *4 = 2^9  = 512
        2:  256,   # 128*2  = 256  = 2^8  *4 = 2^10 = 1024
        4:  512,   # 128*4  = 512  = 2^9  *4 = 2^11 = 2048
        8:  1024,  # 128*8  = 1024 = 2^10 *4 = 2^12 = 4096
        16: 2048,  # 128*16 = 2048 = 2^11 *4 = 2^13 = 8192
        32: 4096,  # 128*32 = 4096 = 2^12 *4 = 2^14 = 16384
        64: 8192,  # 128*64 = 8192 = 2^13 *4 = 2^15 = 32768
    }
    return switcher.get(argument, "nothing")


def prepare_FE_data(inputPath):
    
#     inputPath = 'Comsol_output_Pe_u_Brinkman__Phi2_III__R8192.csv'
    df = pd.read_csv(inputPath, header=4)
    
    title = df.columns.values.tolist()
    
    
    assert title[5]   == 'porosity (1)'
    assert title[6]   == 'Thiele modulus'
    assert title[7]   == 'Peclet number (1)'
    assert title[8]   == 'diffusion ratio of flow to cell phase [0.01-1]'
    
    assert title[9]   == 'normalized intrinsic average concentration in flow domain 1 (1)'
    assert title[10]  == 'normalized intrinsic average concentration in flow domain 2 (1)'
    assert title[11]  == 'normalized intrinsic average concentration in flow domain 3 (1)'
    assert title[12]  == 'normalized intrinsic average concentration in flow domain 4 (1)'
    assert title[13]  == 'normalized intrinsic average concentration in flow domain 5 (1)'
    
    assert title[14]  == 'normalized forward finite difference on intrinsic average concentration in domain 1 (1)'
    assert title[15]  == 'normalized forward finite difference on intrinsic average concentration in domain 2 (1)'
    assert title[16]  == 'normalized center finite difference on intrinsic average concentration in domain 3 (1)'
    assert title[17]  == 'normalized center finite difference on intrinsic average concentration in domain 4 (1)'
    assert title[18]  == 'normalized backward finite difference on intrinsic average concentration in domain 5 (1)'
    
    assert title[19]  == 'Effectiveness factor in domain 1 (1)'
    assert title[20]  == 'Effectiveness factor in domain 2 (1)'
    assert title[21]  == 'Effectiveness factor in domain 3 (1)'
    assert title[22]  == 'Effectiveness factor in domain 4 (1)'
    assert title[23]  == 'Effectiveness factor in domain 5 (1)'
    
    #               epsl          phi2          Pe            Dr            c_f_avg_norm   d_f_c_avg_norm  eta_d_f
    ft1 = np.array([df[title[5]], df[title[6]], df[title[7]], df[title[8]], df[title[9]],  df[title[14]], df[title[19]]]).T
    ft2 = np.array([df[title[5]], df[title[6]], df[title[7]], df[title[8]], df[title[10]], df[title[15]], df[title[20]]]).T
    ft3 = np.array([df[title[5]], df[title[6]], df[title[7]], df[title[8]], df[title[11]], df[title[16]], df[title[21]]]).T
    ft4 = np.array([df[title[5]], df[title[6]], df[title[7]], df[title[8]], df[title[12]], df[title[17]], df[title[22]]]).T
    ft5 = np.array([df[title[5]], df[title[6]], df[title[7]], df[title[8]], df[title[13]], df[title[18]], df[title[23]]]).T
    
    ft = np.concatenate((ft2,ft3,ft4,ft5), axis=0)
    
    return ft


def plot_prepared_data(inputPath, ft):
    df = pd.read_csv(inputPath, header=4)
    title = df.columns.values.tolist()
    
#      hist, edges = np.histogram(eta, bins=bins, range=(0,1), density=False)
    fig, ax = plt.subplots(3, 2, tight_layout=True)

    ax[0][0].hist(ft[:,0]); ax[0][0].set_xlabel(title[5][:-4])
    ax[0][1].hist(ft[:,1]); ax[0][1].set_xlabel(title[6])
    ax[1][0].hist(ft[:,2]); ax[1][0].set_xlabel(title[7][:-4])
    ax[1][1].hist(ft[:,3]); ax[1][1].set_xlabel(title[8])
    ax[2][0].hist(ft[:,4]); ax[2][0].set_xlabel('<c>')
    ax[2][1].hist(ft[:,5]); ax[2][1].set_xlabel('dx<c>')

    fig2, ax2 = plt.subplots(1, 1, tight_layout=True)
    plt.hist(ft[:,-1])
    ax2.set_xlabel('$\eta$')

    
 