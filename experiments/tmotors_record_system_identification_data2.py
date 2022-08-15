import numpy as np

from double_pendulum.controller.trajectory_following.feed_forward import FeedForwardController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment


torque_limit = [4.0, 4.0]

# trajectory
dt = 0.002
t_final = 10.0
N = int(t_final / dt)
T_des = np.linspace(0, t_final, N+1)

# u1 = np.zeros(N+1)

# u1 = 0.4*np.sin(10.*T_des)
# u2 = 0.8*np.cos(10.*T_des)

# u1 = 0.6*np.sin(5.*T_des)
# u2 = 0.4*np.cos(10.*T_des)

u1 = 0.6*np.sin(10.*T_des)
u2 = 0.6*np.cos(5.*T_des)

U_des = np.array([u1, u2]).T

# controller
controller = FeedForwardController(T=T_des,
                                   U=U_des,
                                   torque_limit=torque_limit,
                                   num_break=40)
controller.init()

run_experiment(controller=controller,
               dt=dt,
               t_final=10.,
               can_port="can0",
               motor_ids=[7, 8],
               tau_limit=torque_limit,
               save_dir="data/double-pendulum/tmotors/sysid")
