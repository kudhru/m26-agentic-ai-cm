"""
Sanity-checks the growth/reduction figures quoted on the IndicCorp v2 slides,
using the exact numbers printed in Table 1 and Table 3 of the IndicXTREME
paper (Doddapaneni et al., ACL 2023).
"""

v1_total, v2_total = 8789, 20920  # millions of tokens, Table 3 "Total" row
print(f"IndicCorp v1 -> v2 growth: {v2_total / v1_total:.2f}x  (paper states 2.3x)")

hi_v1, hi_v2 = 1860, 6107  # millions of tokens, Table 3, Hindi row
print(f"Hindi v1 -> v2 growth: {hi_v2 / hi_v1:.2f}x  (paper states 3.3x)")

before_filter, after_filter = 23.1, 20.9  # billions of tokens, Section 4.2
reduction_pct = (before_filter - after_filter) / before_filter * 100
print(f"Offensive-word filtering removes {reduction_pct:.1f}% of tokens "
      f"({before_filter}B -> {after_filter}B)")

indic_lang_tokens, en_tokens, stated_total = 14.4, 6.5, 20.9  # billions, Section 4
print(f"{indic_lang_tokens} + {en_tokens} = {indic_lang_tokens + en_tokens}B "
      f"vs. stated total {stated_total}B (23 Indic languages + Indian English)")
