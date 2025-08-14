- Removed zero-defaulting when loading cache files; now missing fields are left absent instead of set to 0.
- Treat `{total_moves: 0, robots_moved: 0}` as legacy placeholders and load them as empty feature dicts.
- `get_matching_seeds` now skips entries without both `total_moves` and `robots_moved` keys, preventing false matches for levels with `<=` constraints.
- No change to how solved plans are added; per-combination files are only created for real solved feature pairs. 
- Changed `rl_board_games/core/plan_cache.py` to avoid defaulting missing features to zeros when loading cache files.
 