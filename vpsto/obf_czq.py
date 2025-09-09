'''
基函数类
'''                     
import numpy as np

class OBF:
    """Optimal Basis Functions (OBF) - 最优基函数生成器
    用于高效计算时间最优轨迹的基函数及其导数，支持多自由度系统。
    核心功能是通过分段多项式生成平滑的轨迹基函数。

    Attributes:
        ndof (int): 系统的自由度数量(例如机械臂关节数)
        h (np.ndarray): 各时间段长度的数组，形状(N,)
        N (int): 时间段数量
        T (float): 总轨迹时长
        nw (int): 基函数权重向量的标量维度(N + 3)
        M (np.ndarray): 局部段加速度矩阵，形状(N, 2, 2)
        P (np.ndarray): 全局转换矩阵，形状(N+1, nw)
    """
    def __init__(self, ndof):
        """初始化OBF生成器
        Args:
            ndof (int): 系统自由度数量
        """
        self.ndof = ndof # 自由度，可以是ndof=2:(x,y),ndof=3:(q1,q2,q3)
        self.h = None # 时间段序列
        
    def setup(self, h):
        """设置OBF参数
        Args:
            h (np.ndarray): 时间段序列，形状(N,)
        Returns:
            M (np.ndarray): 局部段加速度矩阵，形状(N, 2, 2),即行2列(N*2),N个2*2矩阵
            P (np.ndarray): 全局转换矩阵，形状(N+1, nw_scalar)
        """
        if np.array_equal(h, self.h):
            return
        self.h = h # 时间段序列
        self.N = len(h) # 线段数量
        self.T = np.sum(h) # 总时间
        self.nw = self.N + 1 + 2 # +1:线段数->节点数，+2: 起始节点速度约束数量
        self.t = None
        
        self.M = np.zeros((self.N, 2, 2))#每段轨迹下是一个2*2的矩阵
        for n in range(self.N):
            hn = h[n]
            self.M[n] = 2/(hn**3)*np.array([[6, 3*hn], [3*hn, 2*hn*hn]])# 2*2矩阵,参考公式
        self.cal_P()#只要N不变，P就不变
            
    def get_Lw(self,n):
        """获取第n个基函数的权重矩阵
        Args:
            n (int): 基函数索引
        Returns:
            Lw (np.ndarray): 权重矩阵，形状(2, nw)
        """
        Lw = np.zeros((2, self.nw))
        Lw[0, n] = -1
        Lw[0, n+1] = 1
        return Lw
    
    def get_Lv(self,n):
        """获取第n个基函数的速度矩阵
        Args:
            n (int): 基函数索引
        Returns:
            Lv (np.ndarray): 速度矩阵，形状(2, N+1)
        """
        Lv = np.zeros((2, self.N+1))
        Lv[0, n+1] = -self.h[n]
        Lv[1, n] = -1
        Lv[1, n+1] = 1
        return Lv
    
    def cal_P(self):
        """计算全局转换矩阵
        Returns:
            P (np.ndarray): 全局转换矩阵，形状(N+1, nw)
            Pv (np.ndarray): 速度全局转换矩阵，形状(N+1, N+1)
            Pw (np.ndarray): 权重全局转换矩阵，形状(N+1, nw)
        """
        Pv = np.zeros((self.N+1, self.N+1))
        Pw = np.zeros((self.N+1, self.nw))
        for n in range(self.N-1):
            an = np.array([[0., 1.]]) @ self.M[n+1]
            bn = np.array([[-self.h[n], 1.]]) @ self.M[n]
            Pv[n] = bn @ self.get_Lv(n) - an @ self.get_Lv(n+1)
            Pw[n] = an @ self.get_Lw(n+1) - bn @ self.get_Lw(n) 
        Pv[self.N-1, 0] = 1
        Pw[self.N-1, self.N+1] = 1
        Pv[self.N, self.N] = 1
        Pw[self.N, self.N+2] = 1
        self.Pw = Pw
        self.Pv = Pv
        self.P = np.linalg.inv(Pv) @ Pw
        return self.P
    
    def get_Omega(self, n):
        """获取第n个基函数的Omega矩阵
        Args:
            n (int): 基函数索引
        Returns:
            Omega (np.ndarray): Omega矩阵,形状(2, nw)
        """
        return self.M[n]@(self.get_Lw(n) + self.get_Lv(n) @ self.P)
        #M为2*2矩阵，Lw为2*nw矩阵，Lv为2*N+1矩阵，P为N+1*nw矩阵
        #Omega为2*nw矩阵
    
    def get_base(self, t, der):
        """获取基函数值phi
        Args:
            t (float or np.ndarray): 时间点或时间点序列
            der (int): 导数阶数(0:位置, 1:速度, 2:加速度)
        Returns:
            base_ (np.ndarray): 基函数值矩阵，形状(ndof*N, nw)
        """
        base_ = np.zeros((self.ndof * np.size(t), self.ndof * self.nw))
        #我没假设t是一个标量，base的维度是节点数的列，子矩阵是自由度*自由度
        if np.size(t) == 1:
            t_array = np.array([t])
        else:
            t_array = np.array(t)
        for i in range(np.size(t)):
            t_ = np.max([0., np.min([self.T, t_array[i]])])#确保取到的t不小于0且不大于总时长
            t_start = 0.
            for n in range(self.N):
                if t_ <= t_start + self.h[n] + 1e-6:#1e-6是为了保证数据运算
                    t_ = t_ - t_start
                    q = np.zeros((1, self.nw))
                    q[0, n] = 1.
                    v = np.zeros((1, self.N + 1))
                    v[0, n] = 1.
                    Omega = self.get_Omega(n)
                    if der == 0:
                        base = [-1.0 / 6.0 * t_**3, 1.0 / 2.0 * t_**2] @ Omega + t_ * v @ self.P +q
                    elif der == 1:
                        base = [-1.0 / 2.0 * t_**2, t_] @ Omega + v @ self.P
                    elif der == 2:
                        base = [-t_, 1] @ Omega
                    else:
                        print('Invalid argument for der.')
                    #Omega为2*nw矩阵
                    #base为1*nw矩阵
                    base_[i*self.ndof:(i+1)*self.ndof] = np.kron(base, np.eye(self.ndof))
                    #将数据从1维展开到ndof维度
                    #base的大类i到i+1行是第i个节点的基函数
                    #a,b,c + ndof=2 -> a 0 b 0 c 0
                    #                  0 a 0 b 0 c
                    #t才是生成轨迹段的数量，nw只是用到的轨迹的数量
                    break
                t_start += self.h[n]# 找到每个t_所在的线段
        return base_
    
    def get_y(self, t, y_nodes, v0, vT):#t为评估点的时间序列\
        """获取轨迹值
        Args:
            t (float or np.ndarray): 时间点或时间点序列
            y_nodes (np.ndarray): 节点位置，形状(ndof, N)
            v0 (np.ndarray): 起始点速度，形状(ndof)
            vT (np.ndarray): 终止点速度，形状(ndof)
        Returns:
            y (np.ndarray): 轨迹值，形状(ndof, N)
        """
        self.update_eval(t)
        #y_nodes是节点位置，v0、vT是起始点速度
        w = np.concatenate((y_nodes.flatten(), v0, vT))#生成约束
        #w为(ndof*nw)*1列矩阵，自由度横向展开[x1,y1,x2,y2,x3,y3】
        #base为(t*ndof)*(ndof*nw)矩阵
        y = self.phi_y @ w
        #y为(t*ndof)*1矩阵                    #X1
        #X1=a*x1+b*x2+c*x3                    #Y1
        #Y1=a*y1+b*y2+c*y3                    #X2
        #上面是t=1的情况                        #Y2
        #                               上面是t=2的情况
    
        if np.size(t) == 1:
            return y.reshape(self.ndof)
        return y.reshape(np.size(t), self.ndof)
        #X1 X2 X3
        #Y1 Y2 Y3
        #y的形状如上面所示
    
    def get_v(self, t, y_nodes, v0, vT):
        """获取轨迹速度
        Args:
            t (float or np.ndarray): 时间点或时间点序列
            y_nodes (np.ndarray): 节点位置，形状(ndof, N)
            v0 (np.ndarray): 起始点速度，形状(ndof)
            vT (np.ndarray): 终止点速度，形状(ndof)
        Returns:
            v (np.ndarray): 轨迹速度，形状(ndof, N)
        """
        self.update_eval(t)
        w = np.concatenate((y_nodes.flatten(), v0, vT))#生成约束
        # print("w",w)
        v = self.phi_v @ w
        # print("get_base",self.get_base(t, 1))
        if np.size(t) == 1:
            return v.reshape(self.ndof)
        return v.reshape(np.size(t), self.ndof)
    
    def get_a(self, t, y_nodes, v0, vT):
        """获取轨迹加速度
        Args:
            t (float or np.ndarray): 时间点或时间点序列
            y_nodes (np.ndarray): 节点位置，形状(ndof, N)
            v0 (np.ndarray): 起始点速度，形状(ndof)
            vT (np.ndarray): 终止点速度，形状(ndof)
        Returns:
            a (np.ndarray): 轨迹加速度，形状(ndof, N)
        """
        self.update_eval(t)
        w = np.concatenate((y_nodes.flatten(), v0, vT))#生成约束
        a = self.phi_a @ w
        if np.size(t) == 1:
            return a.reshape(self.ndof)
        return a.reshape(np.size(t), self.ndof)

    def update_eval(self, t):#只要t和N改变，就需要更新
        """更新评估点对应的基函数
        Args:
            t (float or np.ndarray): 时间点或时间点序列
        """
        if np.array_equal(self.t, t):
            return self.phi_y, self.phi_v, self.phi_a
        self.t = t
        self.phi_y = self.get_base(t, 0)
        self.phi_v = self.get_base(t, 1)
        self.phi_a = self.get_base(t, 2)
        return self.phi_y, self.phi_v, self.phi_a
        
    def eval_trajectory(self, t, y_nodes = None, v0 = None, vT = None, w = None):
        """评估轨迹值、速度、加速度
        Args:
            t (float or np.ndarray): 时间点或时间点序列
            y_nodes (np.ndarray): 节点位置，形状(ndof, N)
            v0 (np.ndarray): 起始点速度，形状(ndof)
            vT (np.ndarray): 终止点速度，形状(ndof)
            w (np.ndarray): 轨迹参数，形状(ndof*nw+2*ndof)
        Returns:
            y (np.ndarray): 轨迹值，形状(ndof, N)
            v (np.ndarray): 轨迹速度，形状(ndof, N)
            a (np.ndarray): 轨迹加速度，形状(ndof, N)
        """
        #评估点的时间序列，节点位置，起始点速度，终止点速度
        self.update_eval(t)
        if w is None:
            w = np.concatenate((y_nodes.flatten(), v0, vT))#生成约束
        self.y = self.phi_y @ w
        self.v = self.phi_v @ w
        self.a = self.phi_a @ w
        if np.size(t) == 1:
            return self.y.reshape(self.ndof), self.v.reshape(self.ndof), self.a.reshape(self.ndof)
        return self.y.reshape(np.size(t), self.ndof), self.v.reshape(np.size(t), self.ndof), self.a.reshape(np.size(t), self.ndof)
    
    def get_Phi(self, t):
        """获取轨迹基函数
        Args:
            t (float or np.ndarray): 时间点或时间点序列
        Returns:
            Phi (np.ndarray): 轨迹基函数，形状(ndof, N)
        """
        return self.get_base(t, 0)
    
    def get_dPhi(self, t):
        """获取轨迹速度基函数
        Args:
            t (float or np.ndarray): 时间点或时间点序列
        Returns:
            dPhi (np.ndarray): 轨迹速度基函数，形状(ndof, N)
        """
        return self.get_base(t, 1)
    
    def get_ddPhi(self, t):
        """获取轨迹加速度基函数
        Args:
            t (float or np.ndarray): 时间点或时间点序列
        Returns:
            ddPhi (np.ndarray): 轨迹加速度基函数，形状(ndof, N)
        """
        return self.get_base(t, 2)