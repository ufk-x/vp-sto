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
    env_infos = env_info_gen()
    env = SimpleWorld(use_gui=render, mp4=None)
    env.load_world(env_infos, robot=True)
    env.reset(env_infos)

    # 目标姿态可视化（可选）
    env.render_pose([0.5, 0.0, 0.3])

    # 一些状态缓存
    q_init = env.get_joint_positions(env.robot)           # 当前全部可动关节（含夹爪）
    q_demo = list(q_init)                                 # 可在其上修改并下发
    gripper_open = True

    # 时间调度段落（按秒）
    t0 = time.perf_counter()
    swing_duration = 2.0      # 0-2s: 摆动第4关节
    grip_duration  = 1.0      # 2-3s: 夹爪动作
    done_demo = False

    print("[INFO] Press 'p' to print EEF pose, 'g' to toggle gripper, 'r' to reset, 'ESC' to exit.")

    while p.isConnected():
        t = time.perf_counter() - t0

        # --------- 调度/状态机 ----------
        if not done_demo:
            if t < swing_duration:
                # 让第4关节（索引3）做小幅正弦摆动
                amp = np.deg2rad(15)
                q_demo[3] = q_init[3] + amp * np.sin(2*np.pi * t / swing_duration)
                env.set_joint_positions(env.robot, q_demo, gripper=True)

            elif t < (swing_duration + grip_duration):
                # 切换一次夹爪（索引 -2, -1）
                if gripper_open:
                    q_demo[-2], q_demo[-1] = 0.02, 0.02   # 夹爪合
                else:
                    q_demo[-2], q_demo[-1] = 0.06, 0.06   # 夹爪开
                env.set_joint_positions(env.robot, q_demo, gripper=True)
                gripper_open = not gripper_open
                # 跳到演示结束
                t0 = time.perf_counter() - (swing_duration + grip_duration)
                done_demo = True
            else:
                done_demo = True
        else:
            # 演示完成后的空闲逻辑（例如保持当前位置）
            pass

        # --------- 键盘事件 ----------
        keys = p.getKeyboardEvents()
        if keys:
            # ESC 退出
            if 27 in keys and (keys[27] & p.KEY_WAS_TRIGGERED):
                break
            # 'p' 打印末端位姿
            if ord('p') in keys and (keys[ord('p')] & p.KEY_WAS_TRIGGERED):
                eef_pose = env.get_eef_pose(env.robot)  # [x,y,z, roll,pitch,yaw]
                print("[EEF] pos(xyz), rpy:", np.round(eef_pose, 4))
            # 'g' 切换夹爪
            if ord('g') in keys and (keys[ord('g')] & p.KEY_WAS_TRIGGERED):
                gripper_open = not gripper_open
                q_demo[-2] = 0.06 if gripper_open else 0.02
                q_demo[-1] = q_demo[-2]
                env.set_joint_positions(env.robot, q_demo, gripper=True)
            # 'r' 重置
            if ord('r') in keys and (keys[ord('r')] & p.KEY_WAS_TRIGGERED):
                env.reset(env_infos)
                q_init = env.get_joint_positions(env.robot)
                q_demo = list(q_init)
                gripper_open = True
                t0 = time.perf_counter()
                done_demo = False
                print("[INFO] Reset done.")

        # --------- 物理步进 & 限帧 ----------
        p.stepSimulation()
        time.sleep(env.dt)
