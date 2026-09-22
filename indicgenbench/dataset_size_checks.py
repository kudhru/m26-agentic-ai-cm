"""
Sanity-checks the per-task dataset size figures quoted on the IndicGenBench
dataset-creation slides, using the exact per-language counts and language
counts printed in the paper (Singh et al., ACL 2024).
"""

crosssum_per_lang, n_langs = 700, 29
print(f"CrossSum-IN total: {crosssum_per_lang}*{n_langs} = "
      f"{crosssum_per_lang*n_langs} (paper states 20.3k)")

xquad_passages, xquad_qa, n_xquad_langs = 280, 1390, 12
print(f"XQuAD-IN passages: {xquad_passages}*{n_xquad_langs} = "
      f"{xquad_passages*n_xquad_langs} (paper states 3.3k)")
print(f"XQuAD-IN QA pairs: {xquad_qa}*{n_xquad_langs} = "
      f"{xquad_qa*n_xquad_langs} (paper states 16.6k)")

flores_per_lang, n_flores_langs = 997 + 1012, 29
print(f"FLORES-IN total: {flores_per_lang}*{n_flores_langs} = "
      f"{flores_per_lang*n_flores_langs} (paper states 58.2k)")
