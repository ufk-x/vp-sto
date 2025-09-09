import numpy as np
from .obf import OBF

class VPTraj:
    '''
    轨迹类
    '''
    def __init__(self, ndof, N_eval, N_via, vel_lim, acc_lim):
        '''
        初始化轨迹类
        Args:
            ndof (int): 自由度
            N_eval (int): 轨迹评估点的数量
            N_via (int): 线段的数量
            vel_lim (float): 速度限制
            acc_lim (float): 加速度限制
        '''
        self.ndof = ndof       # 自由度
        self.N_eval = N_eval   # 轨迹评估点的数量
        self.N_via = N_via     # 线段的数量
        self.vel_lim = vel_lim # -vel_lim < dq < vel_lim  
        self.acc_lim = acc_lim # -acc_lim < ddq < acc_lim  
        self.obf = OBF(self.ndof)
        self.__setup_basis()     # 初始化L下三角协方差矩阵
    
    def __setup_basis(self):
        '''
        初始化轨迹基函数
        Function:
            设置轨迹基函数
            固定点不存在优化队列p内,需要剔除,此外,起始点速度也需要剔除
            计算不同情况下的S_chol
        '''
        t_eval = np.linspace(0., 1., self.N_eval)
        self.obf.setup(np.ones(self.N_via)/self.N_via)#传入N，初始化Mn、P
        self.Phi, self.dPhi, self.ddPhi = self.obf.update_eval(t_eval)#传入t，初始化phi、dphi、ddphi
        
        # qT 固定 + dqT 固定
        ddPhi_p = self.ddPhi[:, self.ndof:-3*self.ndof]
        #列代表不同节点，从第一个节点后开始，到倒数第三个节点结束，因为最后3个节点是终点、起终点速度，这些固定不需要优化
        self.S = np.linalg.inv(ddPhi_p.T @ ddPhi_p / self.N_eval) # 1/T * a^2
        self.ddPhi_p_inv = self.S @ ddPhi_p.T / self.N_eval        # 计算 ddPhi_p 的伪逆
        self.S_chol = np.linalg.cholesky(self.S)                  # L
        self.S_chol_inv = np.linalg.inv(self.S_chol)              # L的逆

        # qT 自由 + dqT 固定
        ddPhi_p = self.ddPhi[:, self.ndof:-2*self.ndof]#排除 dqT终点速度，终点速度固定不需要优化
        self.S_q = np.linalg.inv(ddPhi_p.T @ ddPhi_p / self.N_eval) 
        self.ddPhi_p_q_inv = self.S_q @ ddPhi_p.T / self.N_eval         
        self.S_q_chol = np.linalg.cholesky(self.S_q)                
        self.S_q_chol_inv = np.linalg.inv(self.S_q_chol)            

        #qT 固定 + dqT 自由
        ddPhi_p = np.concatenate((self.ddPhi[:, self.ndof:-3*self.ndof],
                                  self.ddPhi[:, -self.ndof:]), axis=1)# 包含 via-points 和 dqT，排除 qT
        self.S_dq = np.linalg.inv(ddPhi_p.T @ ddPhi_p / self.N_eval) 
        self.ddPhi_p_dq_inv = self.S_dq @ ddPhi_p.T / self.N_eval        
        self.S_dq_chol = np.linalg.cholesky(self.S_dq)               
        self.S_dq_chol_inv = np.linalg.inv(self.S_dq_chol)           

        #qT 自由 + dqT 自由
        ddPhi_p = np.concatenate((self.ddPhi[:, self.ndof:-2*self.ndof],
                                  self.ddPhi[:, -self.ndof:]), axis=1)
        self.S_qdq = np.linalg.inv(ddPhi_p.T @ ddPhi_p / self.N_eval) 
        self.ddPhi_p_qdq_inv = self.S_qdq @ ddPhi_p.T / self.N_eval       
        self.S_qdq_chol = np.linalg.cholesky(self.S_qdq)              
        self.S_qdq_chol_inv = np.linalg.inv(self.S_qdq_chol)          


    def get_trajectory(self, p, q0, dq0=None, qT=None, dqT=None, T=None):
        '''
        计算轨迹
        Function:
            计算完整轨迹
            支持批处理多组 via-points 参数
        Args:
            p:via-points 参数，形状为 (batch_size, n_params) 或 (n_params,)
            q0:初始位置,形状 (ndof,)
            dq0:初始速度
            qT:最终位置
            dqT:最终速度
            T:轨迹持续时间，默认为 1.0
        Returns:
            q:位置,形状 (batch_size, N_eval, ndof)
            dq:速度,形状 (batch_size, N_eval, ndof)
            ddq:加速度,形状 (batch_size, N_eval, ndof)
        '''
        if len(p.shape) == 1:
            p = p.reshape(1,-1)#将单组 p 转换为批处理形式
        batch_size = len(p)
        # print("batch_size",batch_size)
        #将 q0, dq0, qT, dqT 扩展为与 batch_size 一致的形状
        q0 = np.tile(q0, (batch_size, 1))#将初始位置q0复制batch_size次，形状为 (batch_size, ndof)
        if dq0 is None:
            dq0 = np.zeros((batch_size, self.ndof))
        else:
            dq0 = np.tile(dq0, (batch_size, 1))
            
        if qT is None and dqT is None: #qT 和 dqT 均自由
            w = np.concatenate((q0, p[:,:-self.ndof], dq0, p[:,-self.ndof:]), axis=1)
        elif qT is None: # qT is contained in p
            dqT = np.tile(dqT, (batch_size, 1))
            w = np.concatenate((q0, p, dq0, dqT), axis=1)#第二好理解的一行， p, qT合并为了p，没有qT就说明包含在p里面
        elif dqT is None: # dqT is contained in p
            qT = np.tile(qT, (batch_size, 1))
            w = np.concatenate((q0, p[:,:-self.ndof], qT, dq0, p[:,-self.ndof:]), axis=1)#dqT 自由 的情况下，p 的最后 ndof 列存储的是 终点速度 dqT
        else: # qT and dqT are given
            qT = np.tile(qT, (batch_size, 1))
            dqT = np.tile(dqT, (batch_size, 1))
            w = np.concatenate((q0, p, qT, dq0, dqT), axis=1)#最好理解的一行，数据按照批次排好
        if T is None:
            T = 1.0

        w[:,-2*self.ndof:] = (T * w[:,-2*self.ndof:].T).T#乘以 T 是为了在后续计算中消去分母的 T

        # Compute the trajectory
        q = (self.Phi @ w.T).T.reshape(batch_size, -1, self.ndof)
        dq = (self.dPhi @ (w.T/T)).T.reshape(batch_size, -1, self.ndof)
        ddq = (self.ddPhi @ (w.T/T**2)).T.reshape(batch_size, -1, self.ndof)
        # print("q.shape",q.shape)
        return q, dq, ddq

    def get_trajectory_at_time(self, t, p, q0, dq0=None, qT=None, dqT=None, T=None):
        '''
        计算特定时间点 t 的位置、速度、加速度
        Function:
            根据输入的 T 实时生成基函数，适用于可变时长轨迹
        Args:
            t:时间点，形状 (batch_size,)
            p:via-points 参数，形状为 (batch_size, n_params) 或 (n_params,)
            q0:初始位置,形状 (ndof,)
            dq0:初始速度
            qT:最终位置
            dqT:最终速度
            T:轨迹持续时间，默认为 1.0
        Returns:
            q:位置,形状 (batch_size, ndof)
            dq:速度,形状 (batch_size, ndof)
            ddq:加速度,形状 (batch_size, ndof)
        '''
        if dq0 is None:
            dq0 = np.zeros(self.ndof)

        if qT is None and dqT is None: 
            w = np.concatenate((q0, p[:-self.ndof], dq0, p[-self.ndof:]))
        elif qT is None: 
            w = np.concatenate((q0, p, dq0, dqT))
        elif dqT is None:
            w = np.concatenate((q0, p[:-self.ndof], qT, dq0, p[-self.ndof:]))
        else:
            w = np.concatenate((q0, p, qT, dq0, dqT))
        if T is None:
            T = 1.0
            
        obf = OBF(self.ndof)
        obf.setup(T * np.ones(self.N_via) / self.N_via)
        q = (obf.get_Phi(t) @ w).reshape(-1,self.ndof)
        dq = (obf.get_dPhi(t) @ w).reshape(-1,self.ndof)
        ddq = (obf.get_ddPhi(t) @ w).reshape(-1,self.ndof)

        return q, dq, ddq

    def get_min_duration(self, p, q0, dq0=None, qT=None, dqT=None):
        '''
        计算满足速度/加速度限制的最小时间 T
        Function:
            计算满足速度/加速度限制的最小时间 T
        Args:
            p:via-points 参数，形状为 (batch_size, n_params) 或 (n_params,)
            q0:初始位置,形状 (ndof,)
            dq0:初始速度
            qT:最终位置
            dqT:最终速度
        Returns:
            T:最小时间,形状 (batch_size,)
        '''
        if len(p.shape) == 1:
            p = p.reshape(1,-1)
        batch_size = len(p)

        q0 = np.tile(q0, (batch_size, 1))
        # Check the input
        if dq0 is None:
            dq0 = np.zeros((batch_size, self.ndof))
        else:
            dq0 = np.tile(dq0, (batch_size, 1))

        #以下处理方式与 get_trajectory 相同，若无qT，在p内约束
        if qT is None and dqT is None: 
            q_via_list = np.concatenate((q0, p[:,:-self.ndof]), axis=1)
            dq_list = np.concatenate((dq0, p[:,-self.ndof:]), axis=1)
        elif qT is None: 
            dqT = np.tile(dqT, (batch_size, 1))
            q_via_list = np.concatenate((q0, p), axis=1)
            dq_list = np.concatenate((dq0, dqT), axis=1)
        elif dqT is None: 
            qT = np.tile(qT, (batch_size, 1))
            q_via_list = np.concatenate((q0, p[:,:-self.ndof], qT), axis=1)
            dq_list = np.concatenate((dq0, p[:,-self.ndof:]), axis=1)
        else: 
            qT = np.tile(qT, (batch_size, 1))
            dqT = np.tile(dqT, (batch_size, 1))
            q_via_list = np.concatenate((q0, p, qT), axis=1)
            dq_list = np.concatenate((dq0, dqT), axis=1)

        dq_q = (q_via_list @ self.dPhi[self.ndof:,:-2*self.ndof].T).reshape(batch_size, -1, self.ndof)
        dq_dq = (dq_list @ self.dPhi[self.ndof:,-2*self.ndof:].T).reshape(batch_size, -1, self.ndof)
        T_dq = np.maximum(np.max(dq_q / (self.vel_lim - dq_dq), axis=(1, 2)),
                          np.max(- dq_q / (self.vel_lim + dq_dq), axis=(1, 2)))
        
        ddq_q = (q_via_list @ self.ddPhi[self.ndof:,:-2*self.ndof].T).reshape(batch_size, -1, self.ndof)
        ddq_dq = (dq_list @ self.ddPhi[self.ndof:,-2*self.ndof:].T).reshape(batch_size, -1, self.ndof)
        T_p = ddq_dq / (2. * self.acc_lim)
        T_ddq = np.maximum(np.max(T_p + np.nan_to_num(np.sqrt(T_p**2 + ddq_q / self.acc_lim), nan=-np.inf), axis=(1, 2)),
                           np.max(-T_p + np.nan_to_num(np.sqrt(T_p**2 - ddq_q / self.acc_lim), nan=-np.inf), axis=(1, 2)))
        return np.maximum(T_dq, T_ddq)

