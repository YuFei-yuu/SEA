# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base.legged_robot import LeggedRobot
from .base.legged_robot_pos import LeggedRobotPos
from .base.legged_robot_pos_dynamic import LeggedRobotPosDynamic
from .base.legged_robot_pos_depth_stairs import LeggedRobotPosDepthStairs
from .base.legged_robot_pos_depth_stairs_passability import (
    LeggedRobotPosDepthStairsPassability,
)
from .base.legged_robot_blind_stair_loco import LeggedRobotBlindStairLoco
from .base.legged_robot_pos_stairs_minimal import LeggedRobotPosStairsMinimal
from .go2.go2_blind_stair_loco_config import (
    Go2BlindStairLocoCfg,
    Go2BlindStairLocoCfgPPO,
    Go2BlindStairForwardFinetuneCfg,
    Go2BlindStairForwardFinetuneCfgPPO,
)
from .go2.go2_stairs_minimal_config import (
    Go2PosStairsMinimalCfg,
    Go2PosStairsMinimalCfgPPO,
)
from .go2.go2_pos_config import (
    Go2PosDynamic1Cfg,
    Go2PosDynamic1CfgPPO,
    Go2PosDynamic2Cfg,
    Go2PosDynamic2CfgPPO,
    Go2PosDynamic3Cfg,
    Go2PosDynamic3CfgPPO,
    Go2PosDynamicComplexCfg,
    Go2PosDynamicComplexCfgPPO,
    Go2PosRoughCfg,
    Go2PosRoughCfgPPO,
    Go2PosSparseStaticCfg,
    Go2PosSparseStaticCfgPPO,
    Go2PosDepthStairsCfg,
    Go2PosDepthStairsCfgPPO,
)
from .go2.go2_depth_passability_config import (
    Go2PosDepthStairsPassabilityCfg,
    Go2PosDepthStairsPassabilityCfgPPO,
)

from legged_gym.utils.task_registry import task_registry


task_registry.register("go2_pos_rough", LeggedRobotPos, Go2PosRoughCfg(), Go2PosRoughCfgPPO())
task_registry.register(
    "go2_blind_stair_loco",
    LeggedRobotBlindStairLoco,
    Go2BlindStairLocoCfg(),
    Go2BlindStairLocoCfgPPO(),
)
task_registry.register(
    "go2_blind_stair_loco_forward_finetune",
    LeggedRobotBlindStairLoco,
    Go2BlindStairForwardFinetuneCfg(),
    Go2BlindStairForwardFinetuneCfgPPO(),
)
task_registry.register(
    "go2_pos_stairs_minimal",
    LeggedRobotPosStairsMinimal,
    Go2PosStairsMinimalCfg(),
    Go2PosStairsMinimalCfgPPO(),
)
task_registry.register("go2_pos_sparse_static", LeggedRobotPosDynamic, Go2PosSparseStaticCfg(), Go2PosSparseStaticCfgPPO())
task_registry.register("go2_pos_dynamic_1", LeggedRobotPosDynamic, Go2PosDynamic1Cfg(), Go2PosDynamic1CfgPPO())
task_registry.register("go2_pos_dynamic_2", LeggedRobotPosDynamic, Go2PosDynamic2Cfg(), Go2PosDynamic2CfgPPO())
task_registry.register("go2_pos_dynamic_3", LeggedRobotPosDynamic, Go2PosDynamic3Cfg(), Go2PosDynamic3CfgPPO())
task_registry.register("go2_pos_dynamic_complex", LeggedRobotPosDynamic, Go2PosDynamicComplexCfg(), Go2PosDynamicComplexCfgPPO())
task_registry.register(
    "go2_pos_depth_stairs",
    LeggedRobotPosDepthStairs,
    Go2PosDepthStairsCfg(),
    Go2PosDepthStairsCfgPPO(),
)
task_registry.register(
    "go2_pos_depth_stairs_passability",
    LeggedRobotPosDepthStairsPassability,
    Go2PosDepthStairsPassabilityCfg(),
    Go2PosDepthStairsPassabilityCfgPPO(),
)
