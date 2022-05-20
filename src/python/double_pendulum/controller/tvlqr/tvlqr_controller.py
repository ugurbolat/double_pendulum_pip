import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.lqr.lqr import lqr
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.utils.csv_trajectory import load_trajectory
from double_pendulum.utils.wrap_angles import wrap_angles_diff

class TVLQRController(AbstractController):
    def __init__(self,
                 mass=[0.5, 0.6],
                 length=[0.3, 0.2],
                 com=[0.3, 0.2],
                 damping=[0.1, 0.1],
                 coulomb_fric=[0.0, 0.0],
                 gravity=9.81,
                 inertia=[None, None],
                 torque_limit=[0.0, 1.0],
                 model_pars=None,
                 csv_path="",
                 read_with="pandas",
                 keys=""
                 ):

        # model parameters
        self.mass = mass
        self.length = length
        self.com = com
        self.damping = damping
        self.cfric = coulomb_fric
        self.gravity = gravity
        self.inertia = inertia
        self.torque_limit = torque_limit

        if model_pars is not None:
            self.mass = model_pars.m
            self.length = model_pars.l
            self.com = model_pars.r
            self.damping = model_pars.b
            self.cfric = model_pars.cf
            self.gravity = model_pars.g
            self.inertia = model_pars.I
            self.Ir = model_pars.Ir
            self.gr = model_pars.gr
            self.torque_limit = model_pars.tl

        self.splant = SymbolicDoublePendulum(
                mass=self.mass,
                length=self.length,
                com=self.com,
                damping=self.damping,
                gravity=self.gravity,
                coulomb_fric=self.cfric,
                inertia=self.inertia,
                torque_limit=self.torque_limit)

        # trajectory
        self.T, self.X, self.U = load_trajectory(
                                    csv_path=csv_path,
                                    read_with=read_with,
                                    with_tau=True,
                                    keys=keys)
        self.max_t = self.T[-1]
        self.dt = self.T[1] - self.T[0]

        # default cost parameters
        self.Q = np.diag([4., 4., 0.1, 0.1])
        self.R = 2*np.eye(1)
        self.Qf = np.diag([4., 4., 0.1, 0.1])

        # initializations
        self.K = []
        # self.k = []

    def set_cost_parameters(self,
                            Q=np.diag([4., 4., 0.1, 0.1]),
                            R=2*np.eye(1),
                            Qf=np.diag([4., 4., 0.1, 0.1])):

        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        self.Qf = np.asarray(Qf)


    def init(self):
        self.K = []
        # self.k = []
        for i in range(len(self.T[:-1])):
            A, B = self.splant.linear_matrices(x0=self.X[i], u0=self.U[i])
            K, S, _ = lqr(A, B, self.Q, self.R)
            self.K.append(K)
        A, B = self.splant.linear_matrices(x0=self.X[-1], u0=self.U[-1])
        K, S, _ = lqr(A, B, self.Qf, self.R)
        self.K.append(K)

    def get_control_output(self, x, t):
        n = int(np.around(min(t, self.max_t) / self.dt))

        x_error = wrap_angles_diff(np.asarray(x) - self.X[n])

        tau = self.U[n] - np.dot(self.K[n], x_error)
        #u = np.squeeze(tau)  # does not work (why?)
        u = [tau[0,0], tau[0,1]]
        #print(t, x_error, self.U[n][1], self.K[n][1], u[1])
        u[0] = np.clip(u[0], -self.torque_limit[0], self.torque_limit[0])
        u[1] = np.clip(u[1], -self.torque_limit[1], self.torque_limit[1])
        return u

    def get_init_trajectory(self):
        return self.T, self.X, self.U