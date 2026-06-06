# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base.legged_robot import LeggedRobot
from .base.legged_robot_pos import LeggedRobotPos
from .base.legged_robot_pos_dynamic import LeggedRobotPosDynamic
from .go2.go2_pos_config import (
    Go2PosDynamic1Cfg,
    Go2PosDynamic1CfgPPO,
    Go2PosDynamic2Cfg,
    Go2PosDynamic2CfgPPO,
    Go2PosDynamic3Cfg,
    Go2PosDynamic3CfgPPO,
    Go2PosRoughCfg,
    Go2PosRoughCfgPPO,
    Go2PosSparseStaticCfg,
    Go2PosSparseStaticCfgPPO,
)

from legged_gym.utils.task_registry import task_registry


task_registry.register("go2_pos_rough", LeggedRobotPos, Go2PosRoughCfg(), Go2PosRoughCfgPPO())
task_registry.register("go2_pos_sparse_static", LeggedRobotPosDynamic, Go2PosSparseStaticCfg(), Go2PosSparseStaticCfgPPO())
task_registry.register("go2_pos_dynamic_1", LeggedRobotPosDynamic, Go2PosDynamic1Cfg(), Go2PosDynamic1CfgPPO())
task_registry.register("go2_pos_dynamic_2", LeggedRobotPosDynamic, Go2PosDynamic2Cfg(), Go2PosDynamic2CfgPPO())
task_registry.register("go2_pos_dynamic_3", LeggedRobotPosDynamic, Go2PosDynamic3Cfg(), Go2PosDynamic3CfgPPO())
