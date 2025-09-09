import sys
from os.path import join, abspath, dirname

def absjoin(*args):
    return abspath(join(*args))

import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = absjoin(dirname(__file__), '..')
BENCHMARK_PATH = absjoin(PROJECT_DIR, 'benchmarks')
PLANNER_PATH = absjoin(PROJECT_DIR, 'planners')

MODEL_PATH = absjoin(PROJECT_DIR, 'models')
PANDA_URDF = absjoin(MODEL_PATH, 'franka_panda/panda.urdf')
XARM_URDF = absjoin(MODEL_PATH, 'xarm/xarm7_with_force.urdf')
TABLE_URDF = absjoin(MODEL_PATH, 'table/table.urdf')
HOOK_ENV_WALL = absjoin(MODEL_PATH, 'wall/hook_env_wall.urdf')
TAPE_URDF = absjoin(MODEL_PATH, 'tape/tape_gs.urdf')
MUGTREE_URDF = absjoin(MODEL_PATH, 'mugtree/mugtree.urdf')
UMBRELLA_URDF = absjoin(MODEL_PATH, 'umbrella_closed/umbrella_closed.urdf')
HANGER_OPEN_URDF = absjoin(MODEL_PATH, 'hanger_open/hanger.urdf')
COLLISION_MODELS_PATH = absjoin(MODEL_PATH, 'collision_models')

sys.path.append(PROJECT_DIR)
