import time
import numpy as np
import pybullet as p
from config import PANDA_URDF, COLLISION_MODELS_PATH
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

    while p.isConnected():
        env.step(sleep=True)  # 单步 + 以 env.dt 休眠
