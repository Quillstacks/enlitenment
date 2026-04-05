# Numerical Stability in Deep Learning Lab
# =========================================
# Preventing NaN, Inf, and mysterious training failures.
# Run each section, observe what breaks and what doesn't.

import numpy as np
from stability import (logsumexp_naive, logsumexp_stable,
					   softmax_naive, softmax_stable,
					   log_softmax_naive, log_softmax_stable,
					   logsumexp_comparison, softmax_comparison,
					   log_softmax_comparison)
from gradient_clipping import (clip_by_value, clip_by_norm,
							   exploding_gradients_demo, clipping_comparison,
							   sgd_with_clipping)
from initialization import (initialization_experiment, xavier_vs_he,
							batch_norm_demo)

np.random.seed(42)


# =================================================================
# 1. LOG-SUM-EXP TRICK
# =================================================================
# Q: What happens with x = [1000, 1001, 1002] naively?
# Q: The answer is ~1002.41, perfectly representable in FP64.
#    The problem is the intermediate computation, not the final answer.

logsumexp_comparison()


# =================================================================
# 2. STABLE SOFTMAX
# =================================================================
# Q: softmax([1000, 1000, 1000]) should be [1/3, 1/3, 1/3] by symmetry.
#    The naive version gives [NaN, NaN, NaN]. Why?

softmax_comparison()


# =================================================================
# 3. LOG-SOFTMAX
# =================================================================
# Q: Cross-entropy loss needs log(softmax(x)).
#    Computing softmax then log loses precision for very confident
#    predictions. The direct formula avoids this.

log_softmax_comparison()


# =================================================================
# 4. GRADIENT EXPLOSION AND CLIPPING
# =================================================================
# Q: With ||W|| = 1.5, gradients grow as 1.5^T through T layers.
#    At T=100, that's a factor of 4 * 10^17!
# Q: Clip by norm preserves direction. Clip by value does not.

exploding_gradients_demo()
clipping_comparison()


# =================================================================
# 5. WEIGHT INITIALIZATION
# =================================================================
# Q: Match your init to your activation:
#    Xavier for tanh/sigmoid, He for ReLU
# Q: BatchNorm rescues bad initialization by normalizing activations

initialization_experiment()
xavier_vs_he()
batch_norm_demo()


# =================================================================
# 6. PUTTING IT ALL TOGETHER
# =================================================================

print(f"\n{'='*70}")
print(f"  Numerical Stability Checklist")
print(f"{'='*70}")
print(f"  1. Use log-sum-exp trick for cross-entropy, attention, softmax")
print(f"  2. Use log_softmax instead of log(softmax) for precision")
print(f"  3. Clip gradients by norm if loss spikes or goes to NaN")
print(f"  4. Match initialization to activation (Xavier/He)")
print(f"  5. Use BatchNorm/LayerNorm to stabilize activations")
print(f"  6. Consider BF16 over FP16 (same range as FP32, no loss scaling)")
print(f"{'='*70}")
print(f"\n  Q: Your training shows loss=0.5 for 1000 steps, then NaN.")
print(f"     What do you check first?")
print(f"     1. Gradient norms -> exploding? Add clipping")
print(f"     2. Activation values -> overflow? Check softmax/exp calls")
print(f"     3. Learning rate -> too large? Reduce or use warmup")
print(f"     4. Loss function -> using log(softmax) instead of log_softmax?")
