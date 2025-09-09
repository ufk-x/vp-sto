      
import numpy as np
import cma
import concurrent.futures

from .obf import OBF
from .vptraj import VPTraj
class VPSTOOptions:
    def __init__(self, ndof):
        self.ndof = ndof
        self.vel_lim = 0.10 * np.ones(ndof)
        self.acc_lim = 1.0 * np.ones(ndof)
        self.traj_duration = None
        self.N_eval = 100
        self.N_via = 5
        self.pop_size = 25
        self.sigma_init = 0.5
        self.max_iter = 1000
        self.CMA_diagonal = False
        self.log = False
        self.verbose = True
        self.multithreading = False
        
        self.candidates = dict()
        self.candidates['T'] = 0.0
        self.candidates['pos'] = None
        self.candidates['vel'] = None
        self.candidates['acc'] = None
        
        self.p_best = None
        self.p_mean = None
        self.c_best = None
        self.T_best = None
        self.w_best = None
        if self.log:
            self.candidates_list = []
            self.loss_list = []
            self.via_mean_list = []
            self.via_best_list = []
        
class VPSTOSolution:
    def __init__(self, options):
        self.opt = options
        self.w_best = None
        self.T_best = None
        self.candidates = dict()
        self.candidates['T'] = 0.0
        self.candidates['pos'] = None
        self.candidates['vel'] = None
        self.candidates['acc'] = None
        
        self.p_best = None
        self.p_mean = None
        self.c_best = None
        self.T_best = None
        self.w_best = None
        self.log = False  
        self.p_samples = None
        self.history = []
        self.history_pos_best = []

        if self.log:
            self.candidates_list = []
            self.loss_list = []
            self.via_mean_list = []
            self.via_best_list = []
            
    def get_posvelacc(self, t):
        if self.w_best is None:
            print("No solution available ")
            return [], [], []
        obf = OBF(self.opt.ndof)
        T = np.max([self.T_best, 1e-3])
        obf.setup(T * np.ones(self.opt.N_via) / self.opt.N_via)
        q = (obf.get_Phi(t) @ self.w_best).reshape(-1,self.opt.ndof)
        dq = (obf.get_dPhi(t) @ self.w_best).reshape(-1,self.opt.ndof)
        ddq = (obf.get_ddPhi(t) @ self.w_best).reshape(-1,self.opt.ndof)
        
        return q, dq, ddq    
class VPSTO():
    def __init__(self, options):
        self.opt = options
        self.vptraj = VPTraj(options.ndof, 
                             options.N_eval, 
                             options.N_via,
                             options.vel_lim,
                             options.acc_lim)
        
        self.p_init = None

    def cma_trajectory(self, loss, q0, qT = None, dq0 = None, dqT = None, T = None,p_init = None, sigma_init = 2.5):
        if dq0 is None:
            dq0 = np.zeros(self.opt.ndof)
        if dqT is None and T is None:
            print('Either T or dqT must be given. Setting dqT to zero.')
            dqT = np.zeros(self.opt.ndof)
        if qT is None and dqT is None:
            dim_x = self.opt.ndof * (self.opt.N_via + 1) #dim表示p的维度
            ddPhi_b = np.concatenate((self.vptraj.ddPhi[:,:self.opt.ndof],
                        self.vptraj.ddPhi[:,-2*self.opt.ndof:-self.opt.ndof]),
                        axis = 1)#计算起始点对路径的影响
            mu_p = -self.vptraj.ddPhi_p_qdq_inv @ ddPhi_b @ np.concatenate((q0,dq0))
            #ddPhi_p_qdq_inv是qT free and dqT free下ddPhi_p的逆
            #将起始点对路径的影响考虑分布到每个位置来抵消影响
            sigma_p_chol = self.vptraj.S_qdq_chol
            sigma_p_chol_inv = self.vptraj.S_qdq_chol_inv
        elif qT is None:
            # qT自由待优化
            dim_x = self.opt.ndof * self.opt.N_via
            ddPhi_b = np.concatenate((self.vptraj.ddPhi[:, :self.opt.ndof],
                                      self.vptraj.ddPhi[:, -2*self.opt.ndof:]), 
                                      axis=1)
            mu_p = - self.vptraj.ddPhi_p_q_inv @ ddPhi_b @ np.concatenate((q0, dq0, dqT))
            sigma_p_chol = self.vptraj.S_q_chol
            sigma_p_chol_inv = self.vptraj.S_q_chol_inv
        elif dqT is None:
            # dqT自由待优化
            dim_x = self.opt.ndof * self.opt.N_via
            ddPhi_b = np.concatenate((self.vptraj.ddPhi[:, :self.opt.ndof],
                                      self.vptraj.ddPhi[:, -3*self.opt.ndof:-self.opt.ndof]), 
                                      axis=1)
            mu_p = - self.vptraj.ddPhi_p_dq_inv @ ddPhi_b @ np.concatenate((q0, qT, dq0))
            sigma_p_chol = self.vptraj.S_dq_chol
            sigma_p_chol_inv = self.vptraj.S_dq_chol_inv
        else:
            # qT and dqT 固定
            dim_x = self.opt.ndof * (self.opt.N_via-1)
            ddPhi_b = np.concatenate((self.vptraj.ddPhi[:, :self.opt.ndof],
                                      self.vptraj.ddPhi[:, -3*self.opt.ndof:]), 
                                      axis=1)
            mu_p = - self.vptraj.ddPhi_p_inv @ ddPhi_b @ np.concatenate((q0, qT, dq0, dqT))
            sigma_p_chol = self.vptraj.S_chol
            sigma_p_chol_inv = self.vptraj.S_chol_inv

        sol = VPSTOSolution(self.opt)
        self.p_init = p_init
        if self.p_init is not None:
            x_init = sigma_p_chol_inv @ (self.p_init - mu_p)#将初始猜测中心化（减去均值）
            self.p_init =  None
        else:
            x_init = np.zeros(dim_x)
        # print(self.opt.sigma_init)
        cmaes = cma.CMAEvolutionStrategy(x_init, self.opt.sigma_init,
            {'CMA_diagonal':self.opt.CMA_diagonal,
             'verbose': -1,
             'CMA_active':True,
             'popsize':self.opt.pop_size,
             'tolfun':1e-6,
             'tolx': 1e-6,             # 添加坐标变化条件
             'tolfunhist': 1e-6})       # 添加历史成本条件        
        i = 0
        sol.c_best = np.inf

        while not cmaes.stop() and i < self.opt.max_iter:
            x_samples = np.array(cmaes.ask())
            p_samples = mu_p + (sigma_p_chol @ x_samples.T).T
            #mu_p通过广播机制自动扩展到(popsize,...)

            if T is None:
                sol.candidates['T'] = self.vptraj.get_min_duration(p_samples, q0, dq0, qT, dqT)
            else:
                sol.candidates['T'] = T * np.ones(self.opt.pop_size)
            (sol.candidates['pos'],
             sol.candidates['vel'],
             sol.candidates['acc']) = self.vptraj.get_trajectory(p_samples, q0, dq0, qT, dqT, sol.candidates['T'])
            
            if self.opt.multithreading is False:
                costs = loss(sol.candidates)
            else:
                costs = self.__loss_multithread(loss, sol)
            cmaes.tell(x_samples, costs)
            
            if np.min(costs) < sol.c_best:
                i_best = np.argmin(costs)
                sol.c_best = costs[i_best]
                sol.p_best = mu_p + sigma_p_chol @ cmaes.result.xbest
                sol.T_best = sol.candidates['T'][i_best]
                
            sol.p_mean = mu_p + sigma_p_chol @ cmaes.result.xfavorite
            sol.history_pos_best.append(sol.candidates['pos'][np.argmin(costs)])
            sol.history.append(sol.candidates['pos'])
            if self.opt.verbose:
                print('# VP-STO iteration:', i, 'Current loss:', sol.c_best, end='\r')
            i += 1

        
        if qT is None and dqT is None:
            sol.w_best = np.concatenate((q0, sol.p_best[:-self.opt.ndof], dq0, sol.p_best[-self.opt.ndof:]))
        elif qT is None:
            sol.w_best = np.concatenate((q0, sol.p_best, dq0, dqT))
        elif dqT is None:
            sol.w_best = np.concatenate((q0, sol.p_best[:-self.opt.ndof], qT, dq0, sol.p_best[-self.opt.ndof:]))
        else:
            sol.w_best = np.concatenate((q0, sol.p_best, qT, dq0, dqT))  
        
        # history_trajs_y, history_trajs_dy, history_trajs_ddy = self.vptraj.get_trajectory(np.array(sol.history), q0, dq0, qT, dqT, sol.T_best)  
        
        if self.opt.verbose:
            print('VP-STO finished after', i, 'iterations with a final loss of', sol.c_best)
        return sol #,history_trajs_y, exploration, warm_start
    
    def sample_trajectories(self, N_traj, q0, dq0=None, qT=None, dqT=None, Q=None, R=None, mu_prior=None, P_prior=None, T=None):
        if dqT is None and T is None:
            print('Either T or dqT must be given. Setting dqT to zero.')
            dqT = np.zeros(self.opt.ndof)

        if R is None:
            R = 1
        if Q is not None and isinstance(Q, (int, float)):
            Q = Q * np.eye(self.opt.ndof)
        if dq0 is None:
            dq0 = np.zeros(self.opt.ndof)
        if qT is None or Q is None:
            mu_qT = np.zeros(self.opt.ndof)
            Q = np.zeros((self.opt.ndof, self.opt.ndof))
        else:
            mu_qT = qT

        if dqT is None:
            dim_p = self.opt.ndof * (self.opt.N_via + 1)
            ddPhi_p = np.concatenate((self.vptraj.ddPhi[:, self.opt.ndof:-2*self.opt.ndof],
                                      self.vptraj.ddPhi[:, -self.opt.ndof:]), axis=1)
            ddPhi_b = np.concatenate((self.vptraj.ddPhi[:, :self.opt.ndof],
                                      self.vptraj.ddPhi[:, -2*self.opt.ndof:-self.opt.ndof]), axis=1)
            P_smooth = ddPhi_p.T @ ddPhi_p * R / self.opt.N_eval
            mu_smooth = - self.vptraj.ddPhi_p_qdq_inv @ ddPhi_b @ np.concatenate((q0, dq0))
            Phi_T = np.concatenate((self.vptraj.Phi[-self.opt.ndof:, self.opt.ndof:-2*self.opt.ndof],
                                    self.vptraj.Phi[-self.opt.ndof:, -self.opt.ndof:]), axis=1)
            P_bias = Phi_T.T @ Q @ Phi_T
            mu_bias = np.tile(mu_qT, self.opt.N_via+1)
        else:
            dim_p = self.opt.ndof * self.opt.N_via
            ddPhi_p = self.vptraj.ddPhi[:, self.opt.ndof:-2*self.opt.ndof]
            ddPhi_b = np.concatenate((self.vptraj.ddPhi[:, :self.opt.ndof],
                                      self.vptraj.ddPhi[:, -2*self.opt.ndof:]), axis=1)
            P_smooth = ddPhi_p.T @ ddPhi_p * R / self.opt.N_eval
            mu_smooth = - self.vptraj.ddPhi_p_q_inv @ ddPhi_b @ np.concatenate((q0, dq0, dqT))

            Phi_T = self.vptraj.Phi[-self.opt.ndof:, self.opt.ndof:-2*self.opt.ndof]
            P_bias = Phi_T.T @ Q @ Phi_T
            mu_bias = np.tile(mu_qT, self.opt.N_via)
        print(np.isnan(dqT))
        if mu_prior is None or len(mu_prior) != dim_p:
            mu_prior = np.zeros(dim_p)
        if P_prior is None:
            P_prior = np.zeros((dim_p, dim_p))
        elif isinstance(P_prior, (int, float)):
            P_prior = P_prior * np.eye(dim_p)

        P_post = P_smooth + P_prior + P_bias
        L_P = np.linalg.cholesky(P_post)
        f_post = np.linalg.solve(L_P, P_smooth @ mu_smooth + P_prior @ mu_prior + P_bias @ mu_bias)

        self.white_noise = np.random.normal(size=(N_traj, dim_p))
        p = np.linalg.solve(L_P.T, (f_post + self.white_noise).T).T 
        if T is None:
            T = self.vptraj.get_min_duration(p, q0, dq0, None, dqT)

        q, dq, ddq = self.vptraj.get_trajectory(p, q0, dq0, None, dqT, T)

        return q, dq, ddq, p, T
    
    
    def __call_loss_multithread(self, loss, candidates, costs, idx):
        costs[idx] = loss(candidates)
    
    def __loss_multithread(self, loss, sol):
        pop_size = len(sol.candidates['T'])
        costs = np.empty(pop_size)
        candidates = []
        for i in range(pop_size):
            candidates.append({'pos': sol.candidates['pos'][i],
                                'vel': sol.candidates['vel'][i],
                                'acc': sol.candidates['acc'][i],
                                'T': [sol.candidates['T'][i]]})
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.opt.pop_size) as executor:
            futures = []
            for i in range(pop_size):
                futures.append(executor.submit(self.__call_loss_multithread, loss, candidates[i], costs, i))
            for future in concurrent.futures.as_completed(futures):
                future.result()
        return costs

    