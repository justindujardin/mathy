from typing import List, Dict, Optional
from ruamel.yaml import YAML
from pathlib import Path
import json

yaml_path = Path(__file__).parent.parent / "mkdocs.yml"
yaml = YAML()


YAMLSection = List[Dict[str, List[Dict[str, str]]]]

mkdocs_yaml = yaml.load(yaml_path)
docs_key = "API Documentation"

docs_toc: YAMLSection = [
    {
        "Core": [
            {"Expressions": "api/core/expressions.md"},
            {"Layout": "api/core/layout.md"},
            {"Parser": "api/core/parser.md"},
            {"Tokenizer": "api/core/tokenizer.md"},
            {"Tree": "api/core/tree.md"},
        ]
    },
    {
        "Envs": [
            {"Binomial Distribute": "api/envs/binomial_distribute.md"},
            {"Complex Simplify": "api/envs/complex_simplify.md"},
            {"Poly Combine In Place": "api/envs/poly_combine_in_place.md"},
            {"Poly Commute Like Terms": "api/envs/poly_commute_like_terms.md"},
            {"Poly Grouping": "api/envs/poly_grouping.md"},
            {"Poly Haystack Like Terms": "api/envs/poly_haystack_like_terms.md"},
            {"Poly Simplify Blockers": "api/envs/poly_simplify_blockers.md"},
            {"Poly Simplify": "api/envs/poly_simplify.md"},
        ]
    },
]


site_nav = mkdocs_yaml["nav"]
for nav_obj in site_nav:
    if docs_key in nav_obj:
        nav_obj[docs_key] = docs_toc
        break

print(nav_obj)
