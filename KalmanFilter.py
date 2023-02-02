# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 23:19:29 2023

@author: Hangyu
"""

import numpy as np
import pandas as pd

class KalmanFilter:
    def __init__(self):
        pass
    
    def KF(self, Z, A, H, state_names, x0, P0, Q, R):
        
        '''
        model equation:
        x_t = A*x_{t-1} + Q
        z_t = H*x_t + R
        
        predict step:
        x_t_hat = A*x_{t-1}
        P_t_hat = A*P_{t-1}*A.T + Q    (P is cov of x)
        
        update step:
        K_t = P_t_hat*H.T*inv(H*P_t_hat*H.T+R)
        x_t = x_t_hat + K_t*(z_t - H*x_t_hat)
        P_t = (I - K_t*H)*P_t_hat        
        '''
        
        n_time = len(Z)
        n_state = len(state_names)
        z = Z
    
        "out initialization"
        x = np.array(np.zeros(shape=(n_time, n_state)))
        x[0]=x0
        x_minus = np.array(np.zeros(shape=(n_time, n_state)))
        x_minus[0] = x0
        
        # cov of x
        P = [0 for i in range(n_time)]
        P[0] = P0
        P_minus = [0 for i in range(n_time)]
        P_minus[0] = P0
        
        # Kalman gains
        K = [0 for i in range(n_time)]
        
        "Kalman Filter"
        for i in range(1,n_time):
            z_t = z[i]
            H_t = H[i].T
            R_t = R
    
            "prediction step"
            x_minus[i] = A.dot(x[i-1].T).T
            P_minus[i] = A.dot(P[i-1]).dot(A.T) + Q
            
            "update step"
            temp = H_t.dot(P_minus[i]).dot(H_t.T) + R_t
            K[i] = P_minus[i].dot(H_t.T) / temp
            P[i] = P_minus[i] - (pd.DataFrame(K[i]) @ pd.DataFrame(H_t).T).values.dot(P_minus[i])
            x[i] = x_minus[i] + K[i].dot(z_t.T-H_t.dot(x_minus[i].T))
        
        x = pd.DataFrame(data=x, index=Z.index, columns=state_names)
        x_minus = pd.DataFrame(data=x_minus, index=Z.index, columns=state_names)
        
        self.x_minus = x_minus
        self.x = x
        self.z = Z
        self.Kalman_gain = K
        self.P = P
        self.P_minus = P_minus
        self.state_names = state_names
        self.A = A
    
class FixIntervalSmooth():
    def __init__(self):
        pass

    def FIS(self, res_KF):
        
        '''
        backward recursion:
        x_{t+1}_hat = A*x_t
        P_{t+1}_hat = A*P_t*A.T + Q
        
        smooth step:
        J_t = P_t*A*inv(P_{t+1}_hat)
        x_t_smooth = x_t + J_t*(m_{t+1}_smooth - m_{t+1}_hat)
        P_t_smooth = P_t - J_t*(P_{t+1}_smooth - P_{t+1}_hat)*J_t.T
        '''
        
        N = len(res_KF.x.index)
        n_state = len(res_KF.x.columns)
        x = np.mat(res_KF.x)
        x_minus = np.mat(res_KF.x_minus)
        P = res_KF.P
        P_minus = res_KF.P_minus
        
        x_sm = np.mat(np.zeros(shape=(N, n_state)))
        x_sm[N-1] = x[N-1]
        
        P_sm = [0 for i in range(N)]
        P_sm[N-1] = P[N-1]
        
        J = [0 for i in range(N)]
        
        for i in reversed(range(N-1)):
            J[i] = P[i].dot(res_KF.A).dot(np.linalg.inv(P_minus[i+1]))
            P_sm[i] = P[i] - J[i].dot(P_minus[i+1]-P_sm[i+1]).dot(J[i].T)
            x_sm[i] = (x[i].T + J[i].dot(x_sm[i+1].T-x_minus[i+1].T)).T
        
        x_sm = pd.DataFrame(data=x_sm, index=res_KF.x.index, columns=res_KF.x.columns)
        
        self.x_sm = x_sm
        self.P_sm = P_sm
        self.z = res_KF.z