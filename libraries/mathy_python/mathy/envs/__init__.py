from .binomial_distribution import MathyBinomialDistributionEnv
from .complex_term_simplification import MathyComplexTermSimplificationEnv
from .polynomial_blockers import MathyPolynomialBlockersEnv
from .polynomial_combine_in_place import MathyPolynomialCombineInPlaceEnv
from .polynomial_commute_like_terms import MathyPolynomialCommuteLikeTermsEnv
from .polynomial_grouping import MathyPolynomialGroupingEnv
from .polynomial_like_terms_haystack import MathyPolynomialLikeTermsHaystackEnv
from .polynomial_simplification import PolySimplify


MATHY_BUILTIN_ENVS = [
    MathyBinomialDistributionEnv,
    MathyComplexTermSimplificationEnv,
    MathyPolynomialBlockersEnv,
    MathyPolynomialCombineInPlaceEnv,
    MathyPolynomialCommuteLikeTermsEnv,
    MathyPolynomialGroupingEnv,
    MathyPolynomialLikeTermsHaystackEnv,
    PolySimplify,
]
