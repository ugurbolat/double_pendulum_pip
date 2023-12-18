import gymnasium as gym
import numpy as np
import math


class CustomEnv(gym.Env):
    def __init__(
        self,
        dynamics_func,
        reward_func,
        terminated_func,
        reset_func,
        obs_space=gym.spaces.Box(
            np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])
        ),
        act_space=gym.spaces.Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
        max_episode_steps=1000,
        scaling = True
    ):
        self.dynamics_func = dynamics_func
        self.reward_func = reward_func
        self.terminated_func = terminated_func
        self.reset_func = reset_func

        self.observation_space = obs_space
        self.action_space = act_space
        self.max_episode_steps = max_episode_steps

        self.observation = self.reset_func()
        self.step_counter = 0
        self.scaling = scaling

    def step(self, action):
        self.observation = self.dynamics_func(self.observation, action, scaling = self.scaling)
        reward = self.reward_func(self.observation, action)
        terminated = self.terminated_func(self.observation)
        info = {}
        truncated = False
        self.step_counter += 1
        if self.step_counter >= self.max_episode_steps:
            truncated = True
            self.step_counter = 0
        return self.observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.observation = self.reset_func()
        self.step_counter = 0
        info = {}
        return self.observation, info

    def render(self, mode="human"):
        pass


class double_pendulum_dynamics_func:
    def __init__(
        self,
        simulator,
        dt=0.01,
        integrator="runge_kutta",
        robot="double_pendulum",
        state_representation=2,
        max_velocity=20.0,
        torque_limit=[5.0, 5.0],
        scaling = True
    ):
        self.simulator = simulator
        self.dt = dt
        self.integrator = integrator
        self.robot = robot
        self.state_representation = state_representation
        self.max_velocity = max_velocity

        self.torque_limit = torque_limit
        self.scaling = scaling

    def __call__(self, state, action,scaling = True):
        if scaling:
            x = self.unscale_state(state)
            u = self.unscale_action(action)
            xn = self.integration(x, u)
            obs = self.normalize_state(xn)
            return np.array(obs, dtype=np.float32)
        else:
            u = self.unscale_action(action)
            xn = self.integration(state, u)
            return np.array(xn, dtype=np.float32)


    def integration(self, x, u):
        if self.integrator == "runge_kutta":
            next_state = np.add(
                x,
                self.dt * self.simulator.runge_integrator(x, self.dt, 0.0, u),
                casting="unsafe",
            )
        elif self.integrator == "euler":
            next_state = np.add(
                x,
                self.dt * self.simulator.euler_integrator(x, self.dt, 0.0, u),
                casting="unsafe",
            )
        return next_state

    def unscale_action(self, action):
        """
        scale the action
        [-1, 1] -> [-limit, +limit]
        """
        if self.robot == "double_pendulum":
            a = [
                float(self.torque_limit[0] * action[0]),
                float(self.torque_limit[1] * action[1]),
            ]
        elif self.robot == "pendubot":
            a = np.array([float(self.torque_limit[0] * action[0]), 0.0])
        elif self.robot == "acrobot":
            a = np.array([0.0, float(self.torque_limit[1] * action[0])])
        return a

    def unscale_state(self, observation):
        """
        scale the state
        [-1, 1] -> [-limit, +limit]
        """
        if self.state_representation == 2:
            x = np.array(
                [
                    observation[0] * np.pi + np.pi,
                    observation[1] * np.pi + np.pi,
                    observation[2] * self.max_velocity,
                    observation[3] * self.max_velocity,
                ]
            )
        elif self.state_representation == 3:
            x = np.array(
                [
                    np.arctan2(observation[0], observation[1]),
                    np.arctan2(observation[2], observation[3]),
                    observation[4] * self.max_velocity,
                    observation[5] * self.max_velocity,
                ]
            )
        return x

    def normalize_state(self, state):
        """
        rescale state:
        [-limit, limit] -> [-1, 1]
        """
        if self.state_representation == 2:
            observation = np.array(
                [
                    (state[0] % (2 * np.pi) - np.pi) / np.pi,
                    (state[1] % (2 * np.pi) - np.pi) / np.pi,
                    np.clip(state[2], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                    np.clip(state[3], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                ]
            )
        elif self.state_representation == 3:
            observation = np.array(
                [
                    np.cos(state[0]),
                    np.sin(state[0]),
                    np.cos(state[1]),
                    np.sin(state[1]),
                    np.clip(state[2], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                    np.clip(state[3], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                ]
            )

        return observation


# TODO create a builder class for DoublePendulum/CustomEnv to generate gym.make("DoublePendulumDFKIRIC{suffix}-v0")

# Currently building a single default environment where the arguments already have default values
# GPT
# def build_default_double_pendulum_env(
#     simulator,
#     dt=0.01,
#     integrator="runge_kutta",
#     robot="double_pendulum",
#     state_representation=2,
#     max_velocity=20.0,
#     torque_limit=[5.0, 5.0],
#     scaling = True
# ):
#     dynamics_func = double_pendulum_dynamics_func(
#         simulator,
#         dt=dt,
#         integrator=integrator,
#         robot=robot,
#         state_representation=state_representation,
#         max_velocity=max_velocity,
#         torque_limit=torque_limit,
#         scaling = scaling
#     )

#     def reward_func(state, action):
#         return 0.0

#     def terminated_func(state):
#         return False

#     def reset_func():
#         return np.array([0.0, 0.0, 0.0, 0.0])

#     env = CustomEnv(
#         dynamics_func,
#         reward_func,
#         terminated_func,
#         reset_func,
#         obs_space=gym.spaces.Box(
#             np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])
#         ),
#         act_space=gym.spaces.Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
#         max_episode_steps=1000,
#         scaling = scaling
#     )

#     return env

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator

def build_default_double_pendulum_pendubot_env() -> CustomEnv:

    torque_limit = [5.0, 0.0]

    # HACK hardcoding the model parameters for now
    # related ymal file in
    # data/system_identification/identified_parameters/design_A.0/model_2.0/model_parameters.yml
    # I1: 0.053470810264216295
    # I2: 0.02392374528789766
    # Ir: 7.659297952841183e-05
    # b1: 0.001
    # b2: 0.001
    # cf1: 0.093
    # cf2: 0.078
    # g: 9.81
    # gr: 6.0
    # l1: 0.3
    # l2: 0.2
    # m1: 0.5593806151425046
    # m2: 0.6043459469186889
    # r1: 0.3
    # r2: 0.18377686083653508
    # tl1: 10.0
    # tl2: 10.0

    mpar_dict = {
        "I1": 0.053470810264216295,
        "I2": 0.02392374528789766,
        "Ir": 7.659297952841183e-05,
        "b1": 0.001,
        "b2": 0.001,
        "cf1": 0.093,
        "cf2": 0.078,
        "g": 9.81,
        "gr": 6.0,
        "l1": 0.3,
        "l2": 0.2,
        "m1": 0.5593806151425046,
        "m2": 0.6043459469186889,
        "r1": 0.3,
        "r2": 0.18377686083653508,
        "tl1": 10.0,
        "tl2": 10.0,
        }

    #mpar = model_parameters(filepath=model_par_path)
    mpar = model_parameters()
    mpar.load_dict(mpar_dict)

    mpar.set_motor_inertia(0.0)
    mpar.set_damping([0.0, 0.0])
    mpar.set_cfric([0.0, 0.0])
    mpar.set_torque_limit(torque_limit)
    dt = 0.002
    integrator = "runge_kutta"

    plant = SymbolicDoublePendulum(model_pars=mpar)
    simulator = Simulator(plant=plant)


    # learning environment parameters
    state_representation = 2
    obs_space = obs_space = gym.spaces.Box(
        np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])
    )
    act_space = gym.spaces.Box(np.array([-1]), np.array([1]))
    max_steps = 100


    dynamics_func = double_pendulum_dynamics_func(
        simulator=simulator,
        dt=dt,
        integrator=integrator,
        robot="pendubot",
        state_representation=state_representation,
    )

    def reward_func(observation, action):
        return -(
            observation[0] ** 2.0
            + (observation[1] + 1.0) * (observation[1] - 1.0)
            + observation[2] ** 2.0
            + observation[3] ** 2.0
            + 0.01 * action[0] ** 2.0
        )


    def terminated_func(observation):
        return False


    def noisy_reset_func():
        rand = np.random.rand(4) * 0.01
        rand[2:] = rand[2:] - 0.05
        observation = [-1.0, -1.0, 0.0, 0.0] + rand
        return np.float32(observation)


    env = CustomEnv(
        dynamics_func=dynamics_func,
        reward_func=reward_func,
        terminated_func=terminated_func,
        reset_func=noisy_reset_func,
        obs_space=obs_space,
        act_space=act_space,
        max_episode_steps=100,
    )

    return env
