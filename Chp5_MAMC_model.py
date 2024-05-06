import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
import pickle
import multiprocessing
import time

def copy_py_file():
    '''复制当前文件到新的文件夹中，并返回新文件夹的序号'''
    current_file_path = os.path.realpath(__file__)
    current_dir = os.path.dirname(current_file_path)
    os.chdir(current_dir)
    current_path = os.getcwd()
    files_and_folders = os.listdir(current_path)
    run_lists = [int(item.replace('run','')) for item in files_and_folders if os.path.isdir(item) and item[0:3] == 'run']
    run_lists.sort()
    run_num = int(run_lists[-1]) if len(run_lists) > 0 else 0
    os.mkdir('run'+str(run_num+1))

    with open('model.py', 'rb') as fr:
        content = fr.read()
    with open('run'+str(run_num+1)+'/model.py', 'wb') as fw:
        fw.write(content)
    fr.close()
    fw.close()
    return run_num+1


class parameters:
    '''参数类 PAR_'''
    def __init__(self):
        # 默认参数
        self.k_cross_membrane = np.array([2.3,
                2.4,
                2.6,
                0])   # CO2,CH4,O2,Ac 跨膜传质系数矩阵   cm/h
        self.Henry_cof = np.array([0.818,
                0.035,
                0.032,
                0])    # CO2,CH4,O2,Ac 亨利溶解度系数矩阵  无量纲
        self.Diffusion_cof= np.array([0.0688,
                0.0662,
                0.0871,
                0.0392])   # CO2,CH4,O2,Ac 水中扩散系数矩阵 cm2/h  
        self.f_Diffusion_biofilm_water = np.array([0.6,
                0.6,
                0.6,
                0.22])   # CO2,CH4,O2,Ac 生物膜中扩散系数比例矩阵 无量纲
        self.L_diffusion = 0.001  # 扩散层厚度  cm
        self.alpha_illumi = 216   # 光强衰减系数 /cm 
        self.Yields = [20.31,         # Y0 藻自养生长的CO2产率   mg/mmol
                1.10,        # Y1 藻自养生长的O2/CO2比例  mmol/mmol
                20.16,        # Y2 藻异养生长的Ac产率  mg/mmol
                0.89,        # Y3 藻异养生长的O2/Ac比例
                1.00,        # Y4 藻异养生长的CO2/Ac比例
                4.65,        # Y5 甲烷菌生长的CH4产率  mg/mmol
                1.01,        # Y6 甲烷菌生长的O2/CH4比例
                0.46,        # Y7 甲烷菌生长的CO2/CH4比例
                0.18]        # Y8 甲烷菌生长的Ac/CH4比例 
        self.kinetic_cof = {'K_aa_I':16.3,         # 藻自养生长的光强半饱和常数 umol/m2/s
            'K_aa_co2':9.8e-7,     # 藻自养生长的CO2半饱和常数 mmol/cm^3
            'K_ah_ac':8.3e-5,       # 藻异养生长的Ac半饱和常数 mmol/cm^3
            'K_ah_o2':7.4e-5,       # 藻异养生长的O2半饱和常数 mmol/cm^3
            'K_m_ch4':3.9e-5,       # 甲烷菌生长的CH4半饱和常数 mmol/cm^3
            'K_m_o2':5.1e-5,         # 甲烷菌生长的O2半饱和常数 mmol/cm^3
            'mu_max_aa':0.046,   # 藻自养生长的最大生长速率 /h
            'mu_max_ah':0.093,   # 藻异养生长的最大生长速率 /h
            'mu_max_m':0.098}     # 甲烷菌生长的最大生长速率 /h 
        self.pH = 6.2      # 反应器pH

    def get_HCO3_CO2(self):
        return 10**(self.pH-7)*4.3
    def get_stoich_cof(self):
        return np.array([[-1/self.Yields[0]/(1+self.get_HCO3_CO2()),self.Yields[4]/self.Yields[2]/(1+self.get_HCO3_CO2()),self.Yields[7]/self.Yields[5]/(1+self.get_HCO3_CO2())],  # CO2产物系数[藻自养，藻异养，甲烷菌]
            [0,0,-1/self.Yields[5]],  # CH4产物系数[藻自养，藻异养，甲烷菌]
            [self.Yields[1]/self.Yields[0],-self.Yields[3]/self.Yields[2],-self.Yields[6]/self.Yields[5]], # O2产物系数[藻自养，藻异养，甲烷菌]
            [0,-1/self.Yields[2],self.Yields[8]/self.Yields[5]]])    # Ac产物系数[藻自养，藻异养，甲烷菌]


class reactors:
    '''反应器类 RET_'''

    def __init__(self):
        self.V_l = 92  # 液相室体积 cm^3
        self.V_g = 136  # 气相室体积 cm^3
        self.A = 40.7  # 反应器底面积cm^2
        self.Q_l = 6  # 液相进料流量 cm^3/h
        self.S_in = np.array([0,      # 进水中CO2浓度 mmol/cm^3
            0,      # 进水中CH4浓度 mmol/cm^3
            0,      # 进水中O2浓度 mmol/cm^3
            0])      # 进水中Ac浓度 mmol/cm^3
        self.N_in = 0.00182  # 进水中氮浓度 mmol/cm^3   25mgN/L
        self.F_scch = 12    # 气相进料流量 scch = cm^3/h
        self.pressure = 1        # 反应器压力 atm
        self.ratio_co2 = 0.3     # CO2初始浓度比例，其余为CH4
        self.I_0 = 80 # 生物膜表面光照强度 umol/m^2/s 
        self.Light_time = 12 # 光照/黑暗周期 h/h

    def get_F_in(self):
        return np.array([self.F_scch * self.pressure * 0.040874* self.ratio_co2, # 进气相CO2物质的量流量 mmol/h
            self.F_scch * self.pressure * 0.040874* (1-self.ratio_co2),      # 进气相CH4物质的量流量 mmol/h
            0,                                                # 进气相O2物质的量流量 mmol/h
            0])                                               # 进气相Ac物质的量流量 mmol/h
    def get_S_g_0(self):
        return np.array([self.pressure*0.040874*self.ratio_co2,        # CO2气相浓度 mmol/cm^3, 298K
                    self.pressure*0.040874*(1-self.ratio_co2),    # CH4气相浓度 mmol/cm^3, 298K
                    0,0])                               # O2,Ac气相浓度 mmol/cm^3, 298K
    def get_S_bl_0(self):
        if self.ratio_co2 == 0.3:
            return np.array([0.00016338,1.44200e-05,0,0])
        else:
            return np.array([0,0,0,0])                     # CO2,CH4,O2,Ac 液相浓度 mmol/cm^3


class biofilms:
    '''生物膜类 BF_'''

    def __init__(self, Lfm, Lfa):
        '''初始化生物膜类, Lfm: 藻生物膜厚度 cm, Lfa: 甲烷菌生物膜厚度 cm'''
        self.Lfm = Lfm
        self.Lfa = Lfa
        self.X_a = 156            # 藻生物膜密度 mg/cm^3
        self.X_m = 189            # MOB生物膜密度 mg/cm^3
        self.N_content_aa = 0.004507      # 藻自养生物膜氮含量 mmol/mg
        self.N_content_ah = 0.007041     # 藻异养生物膜氮含量 mmol/mg
        self.N_content_m = 0.00978      # MOB生物膜氮含量 mmol/mg
    def get_Lf(self):
        '''获取生物膜总厚度 cm'''
        return self.Lfm + self.Lfa
    def generate_z(self):
        '''生成生物膜厚度微元列表 cm'''
        dz_min = min(self.Lfa/5, self.Lfm/5) if self.Lfm > 0 else self.Lfa/10
        z = np.linspace(0, self.get_Lf(), max(30, int(self.get_Lf()/ dz_min)))  # 根据生物膜厚度计算生物膜厚度微元列表
        return z
    def generate_dz(self,z):
        '''生成生物膜厚度微元 cm'''
        return z[1] - z[0]              # 生物膜厚度微元 cm
    def generate_dt(self,dz):
        '''生成生物膜时间微元 h'''
        dt = 1/3600*dz*dz*10000     # Z1模型dt, 时间 h 
        return dt
    def generate_S0(self,z,c=0):
        '''生成生物膜初始浓度矩阵 mmol/cm^3'''
        return np.array([np.ones(len(z)) * c,     # CO2浓度 mmol/cm^3
              np.ones(len(z)) * c,     # CH4浓度 mmol/cm^3
              np.ones(len(z)) * c,     # O2浓度 mmol/cm^3
              np.ones(len(z)) * c])    # Ac浓度 mmol/cm^3


# 定义函数
def illum(z,BF_,RET_,PAR_):
    '''计算光照强度 umol/m^2/s'''
    L_f = BF_.get_Lf()
    I_0 = RET_.I_0
    alpha = PAR_.alpha_illumi
    return I_0 * np.exp(-alpha *(L_f - z))

def dXdt(z,S,BF_,RET_,PAR_):
    '''计算生物膜内生长速率 mmol/cm^3/h'''
    kinetic_cof = PAR_.kinetic_cof
    results = np.zeros(3)
    HCO3_CO2 = PAR_.get_HCO3_CO2()
    for i in range(len(z)):
        if z[i] < BF_.Lfm:
            dXdt_aa = 0
            dXdt_ah = 0
            dXdt_m = kinetic_cof['mu_max_m'] * S[1,i]/ (kinetic_cof['K_m_ch4'] + S[1,i]) * S[2,i] / (kinetic_cof['K_m_o2'] + S[2,i]) * BF_.X_m   # 甲烷菌生长速率
        elif z[i] >= BF_.Lfm and z[i] < BF_.get_Lf():
            I = illum(z[i],BF_,RET_,PAR_)
            dXdt_aa = kinetic_cof['mu_max_aa'] * S[0,i]*(1+HCO3_CO2) / (kinetic_cof['K_aa_co2'] + S[0,i]*(1+HCO3_CO2)) * I / (kinetic_cof['K_aa_I'] + I) * BF_.X_a         # 藻自养生长速率
            dXdt_ah = kinetic_cof['mu_max_ah'] * S[3,i] / (kinetic_cof['K_ah_ac'] + S[3,i]) * S[2,i] / (kinetic_cof['K_ah_o2'] + S[2,i]) * BF_.X_a # 藻异养生长速率
            dXdt_m = 0    # 甲烷菌生长速率
        else:
            dXdt_aa = 0
            dXdt_ah = 0
            dXdt_m = 0
        results = np.vstack([results,[dXdt_aa,dXdt_ah,dXdt_m]])
    return results[1:].T


def flux(z,dz,S,S_bl,S_g,PARs):
    '''计算扩散通量，包括气相跨膜和液相扩散层 mmol/cm^2/h'''
    k = PARs.k_cross_membrane
    H = PARs.Henry_cof
    D = PARs.Diffusion_cof
    f_D = PARs.f_Diffusion_biofilm_water
    L_d = PARs.L_diffusion

    flux_m = k * (S_g * H - S[:,0])
    flux_f = -np.diff(S)/dz * (D*f_D).reshape(4,1) 
    flux_l = - D * (S_bl - S[:,-1]) /L_d
    return np.hstack([flux_m.reshape(4,1),flux_f,flux_l.reshape(4,1)])

def dSdt(dz,dXdt_,flux_,PARs):
    '''计算液相浓度变化 mmol/cm^3/h'''
    stoich_cof = PARs.get_stoich_cof()
    return -np.diff(flux_)/dz + np.matmul(stoich_cof,dXdt_)

def dSldt(flux_,S_bl,RETs):
    '''计算液相浓度变化 mmol/cm^3/h'''
    V_l = RETs.V_l
    A = RETs.A
    Q = RETs.Q_l
    S_in = RETs.S_in
    return (Q * S_in + A * flux_[:,-1] - Q * S_bl) / V_l

def new_Sgdt(dt,flux_,S_g,RETs,PARs):
    '''计算气相浓度变化 mmol/cm^3/h'''
    V_g = RETs.V_g
    A = RETs.A
    F_in = RETs.get_F_in()
    HCO3_CO2 = PARs.get_HCO3_CO2()
    mm = (F_in - A * flux_[:,0]*np.array([1+HCO3_CO2,1,1,1])) * dt + S_g * V_g
    pp = np.sum(mm) / 0.040874 / V_g
    if pp > 1:
        mm_out = mm * (pp - 1)
        F_out = mm_out / dt
        return (mm - mm_out) / V_g, F_out
    else:
        return mm / V_g, np.zeros(4)


def S_interpld(z,S):
    '''插值函数'''
    result = []
    x = np.linspace(0,1,len(S[0]))
    x_new = np.linspace(0,1,len(z))
    for i in range(4):
        f = interp1d(x, S[i], kind='linear')
        result.append(f(x_new))
    return np.array(result)

def list_balance_check(list,threshold=0.000001):
    '''检查列表是否已经稳定'''
    if len (list) < 2:
        return True
    else:
        mudiff = (list[-1]-list[-2])/list[-1]
        mudiff[np.isnan(mudiff)] =0
        return np.all(np.abs(mudiff) < threshold)
        

def Z1_dynamic(BF_,RET_,PAR_,S_bl,S_g,S=False,cycle=False):
    '''Z1模型函数'''
    z = BF_.generate_z()
    dz = BF_.generate_dz(z)
    dt = BF_.generate_dt(dz)
    if  S is False:
        S = BF_.generate_S0(z)
        if cycle is False:
            cycle = 1000000
    else:
        if cycle is False:
            cycle = int(500*len(z))     # 迭代次数，可选，值为0则自动计算

    
    dXdt_list = []  # 创建一个列表来存储dXdt的值
    flux_list = []  # 创建一个列表来存储flux的值
    dSdt_list = []  # 创建一个列表来存储dSdt的值
    S_list = []  # 创建一个列表来存储S的值
    S_bl_list = []  # 创建一个列表来存储S_bl的值
    S_g_list = []  # 创建一个列表来存储S_g的值
    F_out_list = []  # 创建一个列表来存储F_out的值

    if not len(z) == S.shape[1]:
        S_interpld_ = S_interpld(z,S)
        S = S_interpld_

    for cycle_i in tqdm(range(cycle)):
        dXdt_ = dXdt(z,S,BF_,RET_,PAR_)
        flux_ = flux(z,dz,S,S_bl,S_g,PAR_)
        dSdt_ = dSdt(dz,dXdt_,flux_,PAR_)
        dSldt_ = dSldt(flux_,S_bl,RET_)
        new_Sgdt_, F_out = new_Sgdt(dt,flux_,S_g,RET_,PAR_)

        S =np.clip(S + dSdt_ * dt , 0, None)
        S_bl = np.clip(S_bl + dSldt_ * dt, 0, None)
        S_g = np.clip(new_Sgdt_, 0, None)

        dXdt_list.append(dXdt_.copy())
        flux_list.append(flux_.copy())
        dSdt_list.append(dSdt_.copy())
        S_list.append(S.copy())
        S_bl_list.append(S_bl.copy())
        S_g_list.append(S_g.copy())
        F_out_list.append(F_out.copy())
        cycle_i += 1

        if cycle == 1000000 and cycle_i % 10000 == 0:
            if list_balance_check(S_bl_list) and list_balance_check(dXdt_list):
                break

    return {'z':z,'dXdt_list':dXdt_list, 'dSdt_list':dSdt_list, 'S_list':S_list, 'S_bl_list':S_bl_list, 'S_g_list':S_g_list, 'flux_list':flux_list, 'F_out_list':F_out_list}

def save_Z1_result(Z1_dynamic_,file_path='',prefix='coculture_'):
    '''保存Z1模型计算结果'''
    with open(file_path+'/'+prefix+'Z1_result.pickle', 'wb') as f:
        pickle.dump(Z1_dynamic_, f)

def load_Z1_result(file_path='',prefix='coculture_'):
    '''读取Z1模型计算结果'''
    with open(file_path+'/'+prefix+'Z1_result.pickle', 'rb') as f:
        Z1_dynamic_ = pickle.load(f)
    return Z1_dynamic_

def get_balanced_result(Z1_dynamic_):
    '''获取Z1模型稳态结果'''
    z = np.array(Z1_dynamic_['z'])
    S = np.array(Z1_dynamic_['S_list'])[-1]
    S_bl = np.array(Z1_dynamic_['S_bl_list'])[-1]
    S_g = np.array(Z1_dynamic_['S_g_list'])[-1]
    F_out = np.array(Z1_dynamic_['F_out_list'])[-1]
    dXdt_ = np.array(Z1_dynamic_['dXdt_list'])[-1]
    return z, S, S_bl, S_g, F_out, dXdt_


def figure_draw_Z1(Z1_dynamic_,line_num=10):
    '''绘制Z1模型结果（非稳态）'''
    draw_list = ['dXdt_list','S_list','dSdt_list']
    for draw in draw_list:
        num = Z1_dynamic_[draw][0].shape[0]
        fig, axs = plt.subplots(1, num, figsize=(5*num, 5))
        for figure_i in range(num):
            if line_num > 1:
                for SS in Z1_dynamic_[draw][::int(len(Z1_dynamic_[draw])/(line_num-1))]: 
                    if draw == 'dXdt_list':
                        axs[figure_i].plot(Z1_dynamic_['z'][:-1],SS[figure_i][:-1])
                    else:
                        axs[figure_i].plot(Z1_dynamic_['z'],SS[figure_i])
            if draw == 'dXdt_list':
                axs[figure_i].plot(Z1_dynamic_['z'][:-1],Z1_dynamic_[draw][-1][figure_i][:-1],color='black',linewidth=3,linestyle='--')
            else:
                axs[figure_i].plot(Z1_dynamic_['z'],Z1_dynamic_[draw][-1][figure_i],color='black',linewidth=3,linestyle='--')
            axs[figure_i].set_title(['CO2','CH4','O2','Ac'][figure_i] if num==4 else ['AA','AH','M'][figure_i])
        axs[0].set_ylabel(draw)
        plt.xlabel('z')
        plt.show()
        plt.close()
    
    draw_list_2 = ['S_bl_list','S_g_list','F_out_list']
    for draw in draw_list_2:
        for figure_i in range(4):
            plt.plot(np.array(Z1_dynamic_[draw])[:,figure_i])
        plt.legend(['CO2','CH4','O2','Ac'],loc='upper left')
        plt.title(draw)
        plt.show()
        plt.close()

def growth(z,dXdt_,dT,C_N,BF_,RET_):
    '''生物膜生长和氮去除函数'''
    Q_l = RET_.Q_l
    V_l = RET_.V_l
    A = RET_.A
    N_in = RET_.N_in
    X_m = BF_.X_m
    X_a = BF_.X_a
    d_Lfm = 0
    d_Lfah = 0
    d_Lfaa = 0
    for i in range(len(z)-1):
        if z[i] < BF_.Lfm and z[i+1]<=BF_.Lfm:
            d_Lfm += dXdt_[2][i] / BF_.X_m * (z[i+1] - z[i]) * dT
        elif z[i] < BF_.Lfm and z[i+1] > BF_.Lfm:
            d_Lfm += dXdt_[2][i] / BF_.X_m * (BF_.Lfm - z[i]) * dT
            d_Lfaa += dXdt_[0][i+1] / BF_.X_a * (z[i+1] - BF_.Lfm) * dT
            d_Lfah += dXdt_[1][i+1] / BF_.X_a * (z[i+1] - BF_.Lfm) * dT
        elif z[i] >= BF_.Lfm and i < (len(z)-1):
            d_Lfaa += dXdt_[0][i] / BF_.X_a * (z[i+1] - z[i]) * dT
            d_Lfah += dXdt_[1][i] / BF_.X_a * (z[i+1] - z[i]) * dT
    new_Lfm = BF_.Lfm + d_Lfm
    new_Lfa = BF_.Lfa + d_Lfaa + d_Lfah

    N_content_aa = BF_.N_content_aa
    N_content_ah = BF_.N_content_ah
    N_content_m = BF_.N_content_m
    N_removal = (d_Lfaa*N_content_aa*X_a+d_Lfah*N_content_ah*X_a+d_Lfm*N_content_m*X_m)*A
    N_removal_rate = N_removal/(V_l*dT)
    C_N_new = (C_N*V_l + Q_l*N_in*dT - N_removal)/(V_l + Q_l*dT)
    return biofilms(new_Lfm,new_Lfa), C_N_new, N_removal_rate


def T1_dynamic(BF_,RET_,PAR_,runT=24, dT=1):
    '''T1模型函数'''
    T1_result={'T1':[],'I_T1':[],'z_T1':[],'S_T1':[],'S_bl_T1':[],'S_g_T1':[],'F_out_T1':[],'dXdt_T1':[],'BF_T1':[],'C_N_T1':[],'R_N_T1':[]}  # 创建一个字典来存储结果
    
    T1 = np.linspace(0,runT,int(runT/dT)+1)
    T1_result['T1'] = T1
    #I_T1 = np.array([RET_.I_0 if T1[i] % 24 < RET_.Light_time else 0 for i in range(len(T1))])
    I_T1 = [80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    T1_result['I_T1'] = I_T1

    # 初始化
    S_bl_ = RET_.get_S_bl_0()
    S_g_ = RET_.get_S_g_0()
    S_ = False

    for i in range(len(T1)):
        print('-------T:'+str(T1[i])+ ' h-------')
        RET_.I_0 = I_T1[i]
        Z1_dynamic_ = Z1_dynamic(BF_,RET_,PAR_,S_bl_,S_g_,S_)
        #if i == 0:
            #save_Z1_result(Z1_dynamic_,file_path=file_path,prefix=prefix)  
        if i == 0:
            C_N = RET_.N_in
            R_N = 0
        z_, S_, S_bl_, S_g_, F_out_,dXdt_ = get_balanced_result(Z1_dynamic_)
        T1_result['BF_T1'].append(BF_)
        T1_result['z_T1'].append(z_)
        T1_result['S_T1'].append(S_)
        T1_result['S_bl_T1'].append(S_bl_)
        T1_result['S_g_T1'].append(S_g_)
        T1_result['F_out_T1'].append(F_out_)
        T1_result['dXdt_T1'].append(dXdt_)
        T1_result['C_N_T1'].append(C_N)
        T1_result['R_N_T1'].append(R_N)
        BF_, C_N, R_N= growth(z_,dXdt_,dT,C_N,BF_,RET_)
    return T1_result


def figure_draw_T1(T1_result,save=False,file_path='',prefix='coculture_'):
    '''绘制T1模型结果'''
    T1 = T1_result['T1']
    z_list = T1_result['z_T1']
    S_list = T1_result['S_T1']
    dXdt_list = T1_result['dXdt_T1']
    S_bl_list = np.array(T1_result['S_bl_T1'])
    S_g_list = np.array(T1_result['S_g_T1'])
    F_out_list = np.array(T1_result['F_out_T1'])
    Lfm_list = np.array([item.Lfm for item in T1_result['BF_T1']])
    Lfa_list = np.array([item.Lfa for item in T1_result['BF_T1']])
    C_N_list = np.array(T1_result['C_N_T1'])
    R_N_list = np.array(T1_result['R_N_T1'])
    

    fig, axs = plt.subplots(1, 3, figsize=(5*3, 5))
    for figure_i in range(3):
        for tt in range(len(z_list)):
            axs[figure_i].plot(z_list[tt][:-1],dXdt_list[tt][figure_i][:-1])
        axs[figure_i].set_title(['AA','AH','M'][figure_i])
        axs[figure_i].set_xlabel('z')
    axs[0].set_ylabel('dXdt')
    if not save:
        plt.show()
        plt.close()
    else:
        plt.savefig(file_path+'/'+prefix+'dXdt.pdf')
        plt.close()

    fig, axs = plt.subplots(1, 4, figsize=(5*4, 5))
    for figure_i in range(4):
        for tt in range(len(z_list)):
            axs[figure_i].plot(z_list[tt],S_list[tt][figure_i])
        axs[figure_i].set_title(['CO2','CH4','O2','Ac'][figure_i])
        axs[figure_i].set_xlabel('z')
    axs[0].set_ylabel('S')
    if not save:
        plt.show()
        plt.close()
    else:
        plt.savefig(file_path+'/'+prefix+'S.pdf')
        plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(5*3, 5))
    for line_i in range(4):
            axs[0].plot(T1,S_bl_list[:,line_i])
            axs[1].plot(T1,S_g_list[:,line_i])
            axs[2].plot(T1,F_out_list[:,line_i])
    axs[0].legend(['CO2','CH4','O2','Ac'],loc='best')
    axs[1].legend(['CO2','CH4','O2','Ac'],loc='best')
    axs[2].legend(['CO2','CH4','O2','Ac'],loc='best')
    axs[0].set_title('S_bl')
    axs[1].set_title('S_g')
    axs[2].set_title('F_out')
    axs[0].set_xlabel('T')
    axs[1].set_xlabel('T')
    axs[2].set_xlabel('T')
    axs[0].set_ylabel('S_bl')
    axs[1].set_ylabel('S_g')
    axs[2].set_ylabel('F_out')
    if not save:
        plt.show()
        plt.close()
    else:
        plt.savefig(file_path+'/'+prefix+'S_bl__S_g__F_out.pdf')
        plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(5*3, 5))
    axs[0].plot(T1,Lfm_list)
    axs[0].plot(T1,Lfa_list)
    axs[1].plot(T1,C_N_list)
    axs[2].plot(T1[1:],R_N_list[1:])
    axs[0].legend(['L_f_m','L_f_a'],loc='best')
    axs[0].set_title('biofilm growth')
    axs[1].set_title('N in effluent')
    axs[2].set_title('N removal rate')
    axs[0].set_xlabel('T')
    axs[1].set_xlabel('T')
    axs[2].set_xlabel('T')
    axs[0].set_ylabel('thickness')
    axs[1].set_ylabel('N conc.')
    axs[2].set_ylabel('N removal rate')
    if not save:
        plt.show()
        plt.close()
    else:
        plt.savefig(file_path+'/'+prefix+'biofilm_growth_N_out_rate.pdf')
        plt.close()

def figure_draw_3d_T1(T1_result,save=False,file_path='',prefix='coculture_'):
    '''绘制T1模型结果'''
    T1 = T1_result['T1']
    z_list = T1_result['z_T1']
    S_list = T1_result['S_T1']
    dXdt_list = T1_result['dXdt_T1']

    fig = plt.figure(figsize=(5*3, 5))
    for figure_i in range(3):
        ax = fig.add_subplot(1, 3, figure_i+1, projection='3d')
        for tt in range(len(z_list)-1):
            ax.plot([tt]*len(z_list[tt][:-1]), z_list[tt][:-1], dXdt_list[tt][figure_i][:-1])
        ax.set_title(['AA','AH','M'][figure_i])
        ax.set_xlabel('time')
        ax.set_ylabel('z')
        ax.set_zlabel('dXdt')
    if not save:
        plt.show()
        plt.close()
    else:
        plt.savefig(file_path+'/'+prefix+'dXdt_3d.pdf')
        plt.close()

    fig = plt.figure(figsize=(5*4, 5))
    for figure_i in range(4):
        ax = fig.add_subplot(1, 4, figure_i+1, projection='3d')
        for tt in range(len(z_list)-1):
            ax.plot([tt]*len(z_list[tt]), z_list[tt], S_list[tt][figure_i])
        ax.set_title(['CO2','CH4','O2','Ac'][figure_i])
        ax.set_xlabel('time')
        ax.set_ylabel('z')
        ax.set_zlabel('S')
    if not save:
        plt.show()
        plt.close()
    else:
        plt.savefig(file_path+'/'+prefix+'S_3d.pdf')
        plt.close()

def save_T1_result(T1_result,file_path='',prefix='coculture_'):
    '''保存T1模型计算结果'''
    with open(file_path+'/'+prefix+'T1_result.pickle', 'wb') as f:
        pickle.dump(T1_result, f)

def load_T1_result(file_path='',prefix='coculture_'):
    '''读取T1模型计算结果'''
    with open(file_path+'/'+prefix+'T1_result.pickle', 'rb') as f:
        T1_result = pickle.load(f)
    return T1_result


def run_culture(args):
    '''模拟生物膜生长过程'''

    start = time.time()
    BF_ = args[0]
    RET_ = args[1]
    PAR_ = args[2]
    runT = args[3]       # 24
    dT = args[4]         # 1
    save = args[5]       # True or False
    file_path = args[6] # string
    prefix = args[7]   #string

    T1_result = T1_dynamic(BF_,RET_,PAR_,runT,dT)
    save_T1_result(T1_result,file_path,prefix)
    figure_draw_T1(T1_result,save,file_path,prefix)
    figure_draw_3d_T1(T1_result,save,file_path,prefix)
    #figure_draw_T1(T1_result)
    end = time.time()
    print(prefix+' 运行时间为: %.1f Seconds'%(end-start))
    #Z1_dynamic_ = load_Z1_result(file_path,prefix)
    #figure_draw_Z1(Z1_dynamic_)


if __name__ == '__main__':   
    file_path = 'run'+str(copy_py_file())
    '''    
    def gen_arg_list():
        #生成参数列表
        arg_list = []
        for i in [6,12,18,24]:
            PAR_ = parameters()
            RET_ = reactors()
            RET_.Light_time = i
            BF_ = biofilms(0.001,0.001)
            args=[BF_,RET_,PAR_,480,1,True,file_path,'coculture_Light_time_'+str(i)+'_']
            arg_list.append(args)
        
        for i in [6,12,18,24]:
            PAR_ = parameters()
            RET_ = reactors()
            RET_.Light_time = i
            BF_ = biofilms(0,0.001)
            args=[BF_,RET_,PAR_,480,1,True,file_path,'singlealgae_Light_time_'+str(i)+'_']
            arg_list.append(args)
        
        for i in [4,12,20,30]:
            PAR_ = parameters()
            RET_ = reactors()
            RET_.Q_l = i
            BF_ = biofilms(0.001,0.001)
            args=[BF_,RET_,PAR_,480,1,True,file_path,'coculture_Q_l_'+str(i)+'_']
            arg_list.append(args)
        
        for i in [4,12,20,30]:
            PAR_ = parameters()
            RET_ = reactors()
            RET_.Q_l = i
            BF_ = biofilms(0,0.001)
            args=[BF_,RET_,PAR_,480,1,True,file_path,'singlealgae_Q_l_'+str(i)+'_']
            arg_list.append(args)

        for i in [0.0005,0.02]:
            PAR_ = parameters()
            RET_ = reactors()
            BF_ = biofilms(i,0.001)
            args=[BF_,RET_,PAR_,480,1,True,file_path,'coculture_Lfm_'+str(i)+'_']
            arg_list.append(args)
        
        for i in [0.0005,0.02]:
            PAR_ = parameters()
            RET_ = reactors()
            BF_ = biofilms(0.001,i)
            args=[BF_,RET_,PAR_,480,1,True,file_path,'coculture_Lfa_'+str(i)+'_']
            arg_list.append(args)

        return arg_list
    
    arg_list=gen_arg_list()

    ''' 
    # 自定义参数列表
    PAR_1 = parameters()  # 参数类
    RET_1 = reactors()  # 反应器类
    BF_1 = biofilms(0.001,0.001)  # 生物膜类
    args_1=[BF_1,RET_1,PAR_1,290,1,True,file_path,'coculture_']
    
    PAR_2 = parameters()  # 参数类
    RET_2 = reactors()  # 反应器类
    BF_2 = biofilms(0,0.001)  # 生物膜类
    args_2=[BF_2,RET_2,PAR_2,290,1,True,file_path,'singlealgae_']
    arg_list = [args_1]


    with multiprocessing.Pool() as process_pool:
        process_pool.map(run_culture,arg_list)
  

    # figure_draw_Z1(Z1)


# cd /seu_share/home/zhuguangcan/230189434/MAMC_MfBR_model
# bsub < python_cpu.sh
# cd C:\Users\lix\Documents\PythonScript\MAMC_MfBR_model
# python model.py
