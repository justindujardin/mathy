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

from .base_ops import assert_rank_and_shape_compatibility, best_effort_shape
from .discrete_policy_gradient_ops import (
    discrete_policy_entropy_loss,
    discrete_policy_gradient,
    discrete_policy_gradient_loss,
    sequence_advantage_actor_critic_loss,
)
from .sequence_ops import multistep_forward_view, scan_discounted_sum
from .value_ops import generalized_lambda_returns, td_lambda, td_learning

__version__ = "1.33.7"
