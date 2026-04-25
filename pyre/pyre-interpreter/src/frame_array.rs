// Post-L1: `PyObjectArray` removed from `pyre-object`. `W_List` /
// `W_Tuple` / `DictStorage` now hold `*mut ItemsBlock` directly
// (rlist.py:116 `(length, items)` shape). This module is retained
// as an empty stub for source-compatibility; no re-exports needed.
