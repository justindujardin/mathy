from .binomial_distribute import BinomialDistribute
from .complex_simplify import ComplexSimplify
from .poly_simplify_blockers import PolySimplifyBlockers
from .poly_combine_in_place import PolyCombineInPlace
from .poly_commute_like_terms import PolyCommuteLikeTerms
from .poly_grouping import PolyGroupLikeTerms
from .poly_haystack_like_terms import PolyHaystackLikeTerms
from .poly_simplify import PolySimplify


MATHY_BUILTIN_ENVS = [
    BinomialDistribute,
    ComplexSimplify,
    PolySimplifyBlockers,
    PolyCombineInPlace,
    PolyCommuteLikeTerms,
    PolyGroupLikeTerms,
    PolyHaystackLikeTerms,
    PolySimplify,
]
