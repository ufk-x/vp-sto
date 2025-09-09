import time
import numpy as np
import pybullet as p
from config import PANDA_URDF
from benchmarks.playground.env_simple_world import SimpleWorld

def env_info_gen():
    robots_info = {
        "panda": {
            "urdf": PANDA_URDF,
            "base_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "conf": [0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4, 0.06, 0.06],
            "scale": 1,
        },
    }
    return {"robots_info": robots_info, "utilities": {}}


if __name__ == "__main__":
    render = True
    mp4 = None
    env_infos = env_info_gen()
    env = SimpleWorld(use_gui=render, mp4=mp4)
    env.load_world(env_infos, robot=True)
    env.reset(env_infos)

    # 加三个动态球
    env.add_ball("ball1", pos=[0.5, 0.0, 0.5], radius=0.03, mass=0.0,  # 质量为0,可以模拟假想的动态球
                    color=(1, 0, 0, 1), lin_vel=[0.0, 0.2, 0.0], restitution=0.2)
    env.add_ball("ball2", pos=[0.4, 0.2, 0.65], radius=0.04, mass=0.1,  # 质量不为0,会受重力影响
                    color=(0, 1, 0, 1), lin_vel=[0.0, -0.1, 0.0], restitution=0.4)
    env.add_ball("ball3", pos=[0.3, -0.2, 0.75], radius=0.025, mass=0.03,
                    color=(0, 0, 1, 1), lin_vel=[0.1, 0.0, 0.0], restitution=0.6)
    
    while p.isConnected():
        env.step(sleep=True)  # 单步 + 以 env.dt 休眠
