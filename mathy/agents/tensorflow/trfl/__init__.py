# Copyright 2018 The trfl Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Flattened namespace for ."""

from .action_value_ops import double_qlearning
from .action_value_ops import persistent_qlearning
from .action_value_ops import qlambda
from .action_value_ops import qlearning
from .action_value_ops import qv_learning
from .action_value_ops import sarsa
from .action_value_ops import sarse
from .base_ops import assert_rank_and_shape_compatibility
from .base_ops import best_effort_shape
from .clipping_ops import huber_loss
from .discrete_policy_gradient_ops import discrete_policy_entropy_loss
from .discrete_policy_gradient_ops import discrete_policy_gradient
from .discrete_policy_gradient_ops import discrete_policy_gradient_loss
from .discrete_policy_gradient_ops import sequence_advantage_actor_critic_loss
from .dist_value_ops import categorical_dist_double_qlearning
from .dist_value_ops import categorical_dist_qlearning
from .dist_value_ops import categorical_dist_td_learning
from .dpg_ops import dpg
from .indexing_ops import batched_index
from .periodic_ops import periodically
from .policy_gradient_ops import policy_entropy_loss
from .policy_gradient_ops import policy_gradient
from .policy_gradient_ops import policy_gradient_loss
from .policy_gradient_ops import sequence_a2c_loss
from .retrace_ops import retrace
from .retrace_ops import retrace_core
from .sequence_ops import multistep_forward_view
from .sequence_ops import scan_discounted_sum
from .target_update_ops import periodic_target_update
from .target_update_ops import update_target_variables
from .value_ops import generalized_lambda_returns
from .value_ops import qv_max
from .value_ops import td_lambda
from .value_ops import td_learning
from .vtrace_ops import vtrace_from_importance_weights
from .vtrace_ops import vtrace_from_logits

__version__ = "1.33.7"
