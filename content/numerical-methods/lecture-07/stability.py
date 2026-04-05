# Numerical Stability: Log-Sum-Exp, Stable Softmax, Log-Softmax
#
# Mathematically equivalent formulas can have vastly different
# floating-point behavior.  This script demonstrates the tricks
# that every deep learning framework uses under the hood.
#
# Key identity (log-sum-exp trick):
#   log(sum(e^x_i)) = m + log(sum(e^(x_i - m)))   where m = max(x)
#
# Key property (softmax shift invariance):
#   softmax(x) = softmax(x - c)  for any constant c

import time
import numpy as np


# =====================================================================
# Log-Sum-Exp
# =====================================================================

def logsumexp_naive(x):
	"""
	Naive log-sum-exp: log(sum(exp(x))).

	Fails when any x_i > ~709 (FP64 overflow) or when all x_i are
	very negative (underflow to 0, then log(0) = -inf).
	"""
	return np.log(np.sum(np.exp(x)))


def logsumexp_stable(x):
	"""
	Stable log-sum-exp using the max-shift trick.

	log(sum(e^x_i)) = m + log(sum(e^(x_i - m)))

	With m = max(x):
	  - Largest exponent becomes e^0 = 1   (no overflow)
	  - All exponents are <= 0             (bounded by 1)
	  - Sum is >= 1                        (no underflow to 0)
	"""
	m = np.max(x)
	return m + np.log(np.sum(np.exp(x - m)))


def logsumexp_comparison():
	"""Compare naive vs stable log-sum-exp on various inputs."""
	print(f"\n{'='*70}")
	print(f"Log-Sum-Exp: Naive vs Stable")
	print(f"{'='*70}")
	print(f"{'Input':<30} {'Naive':>14} {'Stable':>14}")
	print(f"{'-'*70}")

	test_cases = [
		([1, 2, 3],               "[1, 2, 3]"),
		([100, 101, 102],         "[100, 101, 102]"),
		([1000, 1001, 1002],      "[1000, 1001, 1002]"),
		([-1000, -999, -998],     "[-1000, -999, -998]"),
		([0, 0, 0],               "[0, 0, 0]"),
		([-100, 0, 100],          "[-100, 0, 100]"),
	]

	for values, label in test_cases:
		x = np.array(values, dtype=np.float64)
		naive = logsumexp_naive(x)
		stable = logsumexp_stable(x)
		print(f"{label:<30} {naive:>14.4f} {stable:>14.4f}")
		time.sleep(0.4)

	# Q: For which inputs does the naive version fail?
	# Q: The stable version always gives the correct answer. Why?
	# Q: What is the computational overhead of the stable version?
	#    (Just one extra max() and subtraction)


# =====================================================================
# Softmax
# =====================================================================

def softmax_naive(x):
	"""
	Naive softmax: e^x_i / sum(e^x_j).

	Fails for large x (overflow) and produces NaN for Inf/Inf.
	"""
	e_x = np.exp(x)
	return e_x / np.sum(e_x)


def softmax_stable(x):
	"""
	Stable softmax using shift invariance.

	softmax(x) = softmax(x - m)  where m = max(x)

	After shifting: largest exponent is e^0 = 1, all others < 1.
	No overflow possible.
	"""
	m = np.max(x)
	e_x = np.exp(x - m)
	return e_x / np.sum(e_x)


def softmax_comparison():
	"""Compare naive vs stable softmax on various inputs."""
	print(f"\n{'='*70}")
	print(f"Softmax: Naive vs Stable")
	print(f"{'='*70}")

	test_cases = [
		([1, 2, 3],           "[1, 2, 3]"),
		([1000, 1001, 1002],  "[1000, 1001, 1002]"),
		([1000, 1000, 1000],  "[1000, 1000, 1000]"),
		([-1000, -999, -998], "[-1000, -999, -998]"),
	]

	for values, label in test_cases:
		x = np.array(values, dtype=np.float64)
		naive = softmax_naive(x)
		stable = softmax_stable(x)

		print(f"\n  Input: {label}")
		print(f"    Naive:  {naive}   sum={np.sum(naive):.4f}")
		print(f"    Stable: {stable}   sum={np.sum(stable):.4f}")
		time.sleep(0.4)

	# Q: For [1000, 1000, 1000], what should softmax give? (Hint: symmetry)
	# Q: The naive version gives NaN even though the answer is [1/3, 1/3, 1/3]


# =====================================================================
# Log-Softmax
# =====================================================================

def log_softmax_naive(x):
	"""
	Naive log-softmax: log(softmax(x)).

	Two problems:
	1. softmax can overflow/underflow
	2. log of very small softmax values loses precision
	"""
	return np.log(softmax_naive(x))


def log_softmax_stable(x):
	"""
	Stable log-softmax computed directly.

	log(softmax(x))_i = x_i - log(sum(e^x_j))
	                   = x_i - m - log(sum(e^(x_j - m)))

	This avoids both overflow and the precision loss from
	computing softmax then taking log.
	"""
	m = np.max(x)
	return x - m - np.log(np.sum(np.exp(x - m)))


def log_softmax_comparison():
	"""Compare naive vs stable log-softmax."""
	print(f"\n{'='*70}")
	print(f"Log-Softmax: Naive vs Stable")
	print(f"{'='*70}")

	test_cases = [
		([1, 2, 3],           "[1, 2, 3]"),
		([1000, 1001, 1002],  "[1000, 1001, 1002]"),
		([-1000, -999, -998], "[-1000, -999, -998]"),
		([0, 0, 0, 0, 0, 0, 0, 0, 0, 100],  "[0,...,0,100] (10 elements)"),
	]

	for values, label in test_cases:
		x = np.array(values, dtype=np.float64)
		naive = log_softmax_naive(x)
		stable = log_softmax_stable(x)

		print(f"\n  Input: {label}")
		print(f"    Naive:  {naive}")
		print(f"    Stable: {stable}")
		time.sleep(0.4)

	# Q: log_softmax is used in cross-entropy loss:
	#    loss = -log(softmax(x))_y  where y is the true class
	# Q: Why does torch.nn.functional.log_softmax exist as a separate function
	#    instead of just doing log(softmax(x))?


if __name__ == "__main__":
	logsumexp_comparison()
	softmax_comparison()
	log_softmax_comparison()
