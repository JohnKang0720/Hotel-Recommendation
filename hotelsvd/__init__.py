"""hotelsvd — matrix factorization for hotel recommendations, from first principles.

Two factorizations live here:

* ``svd``  — a from-scratch SVD via power iteration + deflation (the original
  idea from my notebook, cleaned up and made correct).
* ``funk`` — FunkSVD, which learns latent factors by gradient descent over the
  *observed* ratings only. This is the one that fixes the "treat every missing
  rating as a real 0" bug that wrecked the original results.
"""

__all__ = ["data", "svd", "funk", "baselines", "evaluate", "recommend", "viz"]
