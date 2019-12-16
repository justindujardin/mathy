## Overview

Mathy uses a model that predicts which action to take in an environment, and the scalar value of the current state.

Mathy's policy/value model takes in a [window of observations](/api/state/#mathywindowobservation) and outputs a weighted distribution over all the possible actions and value estimates for each observation.
