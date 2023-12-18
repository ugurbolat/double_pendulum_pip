from gymnasium.envs.registration import register

#from qturtle.builders.default import create_qturtle_env_3x3
from double_pendulum.simulation.gym_env import build_default_double_pendulum_pendubot_env

register(
     id="DoublePendulumDFKIRICPendubot-v0",
     #entry_point="double_pendulum.simulation.gym_env:CustomEnv",
     entry_point=build_default_double_pendulum_pendubot_env,
     max_episode_steps=50, # TODO atm this is not used but hardcoded in the env
)

print("Registered DoublePendulumDFKIRICPendubot-v0")
