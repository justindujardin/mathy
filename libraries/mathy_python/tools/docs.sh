#!/bin/bash
set -e

. .env/bin/activate


# Core
pydocmd simple mathy.core.expressions++ > ../website/docs/api/core/expressions.md
pydocmd simple mathy.core.layout++ > ../website/docs/api/core/layout.md
pydocmd simple mathy.core.parser++ > ../website/docs/api/core/parser.md
pydocmd simple mathy.core.tokenizer++ > ../website/docs/api/core/tokenizer.md
pydocmd simple mathy.core.tree++ > ../website/docs/api/core/tree.md

# Envs
pydocmd simple mathy.envs.binomial_distribute++ > ../website/docs/api/envs/binomial_distribute.md
pydocmd simple mathy.envs.complex_simplify++ > ../website/docs/api/envs/complex_simplify.md
pydocmd simple mathy.envs.poly_combine_in_place++ > ../website/docs/api/envs/poly_combine_in_place.md
pydocmd simple mathy.envs.poly_commute_like_terms++ > ../website/docs/api/envs/poly_commute_like_terms.md
pydocmd simple mathy.envs.poly_grouping++ > ../website/docs/api/envs/poly_grouping.md
pydocmd simple mathy.envs.poly_haystack_like_terms++ > ../website/docs/api/envs/poly_haystack_like_terms.md
pydocmd simple mathy.envs.poly_simplify_blockers++ > ../website/docs/api/envs/poly_simplify_blockers.md
pydocmd simple mathy.envs.poly_simplify++ > ../website/docs/api/envs/poly_simplify.md

