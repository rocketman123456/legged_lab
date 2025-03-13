from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseEnvCfg, BaseAgentCfg
from legged_lab.envs.h1.h1_config import H1FlatEnvCfg, H1RoughEnvCfg, H1FlatAgentCfg, H1RoughAgentCfg
from legged_lab.envs.g1.g1_config import G1FlatEnvCfg, G1RoughEnvCfg, G1FlatAgentCfg, G1RoughAgentCfg
from legged_lab.envs.gr2.gr2_config import GR2FlatEnvCfg, GR2RoughEnvCfg, GR2FlatAgentCfg, GR2RoughAgentCfg
from legged_lab.envs.anymal_d.anymal_d_config import AnymalDFlatEnvCfg, AnymalDRoughEnvCfg, AnymalDFlatAgentCfg, AnymalDRoughAgentCfg
from legged_lab.utils.task_registry import task_registry

from legged_lab.envs.srobot.srobot_biped_config import SrobotBipedFlatEnvCfg, SrobotBipedRoughEnvCfg, SrobotBipedFlatAgentCfg, SrobotBipedRoughAgentCfg

task_registry.register("h1_flat", BaseEnv, H1FlatEnvCfg(), H1FlatAgentCfg())
task_registry.register("h1_rough", BaseEnv, H1RoughEnvCfg(), H1RoughAgentCfg())
task_registry.register("g1_flat", BaseEnv, G1FlatEnvCfg(), G1FlatAgentCfg())
task_registry.register("g1_rough", BaseEnv, G1RoughEnvCfg(), G1RoughAgentCfg())
task_registry.register("gr2_flat", BaseEnv, GR2FlatEnvCfg(), GR2FlatAgentCfg())
task_registry.register("gr2_rough", BaseEnv, GR2RoughEnvCfg(), GR2RoughAgentCfg())
task_registry.register("anymal_d_flat", BaseEnv, AnymalDFlatEnvCfg(), AnymalDFlatAgentCfg())
task_registry.register("anymal_d_rough", BaseEnv, AnymalDRoughEnvCfg(), AnymalDRoughAgentCfg())
task_registry.register("srobot_biped_flat", BaseEnv, SrobotBipedFlatEnvCfg(), SrobotBipedFlatAgentCfg())
task_registry.register("srobot_biped_rough", BaseEnv, SrobotBipedRoughEnvCfg(), SrobotBipedRoughAgentCfg())
