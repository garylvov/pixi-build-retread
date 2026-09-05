# p6ad — gate split predicted BEFORE the run

Written 2026-09-04 21:5x EDT, before `cargo test --lib` was executed on this branch.

Branch `fix/p6ad-repodata-universe-provenance` off `integration/4.12` = `1e8a06e`.
(B9b `1273d62` had NOT landed at branch time: `mBB-relock` 5833380 RUNNING,
`mB9b-land` 5833382 PENDING on its dependency — the brief's fallback applies.)

Baseline at `1e8a06e`: **1738 passed / 0 failed / 21 ignored** (p6ac recorded its own
gate as 1742 = "1738 + 4").

p6ad adds FIVE `#[test]` functions, all in `src/repodata.rs`'s existing `mod tests`:

1. `p6ad_identical_repodata_bytes_are_the_same_universe_whatever_the_mtime_or_path`
2. `p6ad_one_changed_byte_in_one_subdir_moves_the_universe_digest`
3. `p6ad_the_universe_digest_is_keyed_per_channel_and_subdir_not_on_a_bag_of_hashes`
4. `p6ad_a_sidecar_that_does_not_describe_the_file_is_ignored_not_believed`
5. `p6ad_the_summary_names_every_document_the_snapshot_actually_holds`
6. `p6ad_an_empty_universe_still_has_a_digest`

That is SIX, not five — counted off the file, not off memory.

**PREDICTED SPLIT: 1744 passed / 0 failed / 21 ignored.**

No test is deleted, renamed or ignored. The `AdvertisedIdentityRecord` SCHEMA bump
(3 -> 4) and the new `describes()` argument change four existing call sites and no
test's assertion, so no existing test's outcome should move.

---

## AMENDMENT, written 2026-09-04 ~22:30 EDT, BEFORE the re-run

**The tip moved under me and the branch was rebased.** `mB9b-land` 5833382 completed
while p6ad was being built, so `integration/4.12` is now **`1273d62`** (= `1e8a06e` +
`f59aa35`, the p6z auto-detect line). Two things forced the rebase rather than made it
optional:

1. the brief says use `1273d62` if B9b has landed — it now has; and
2. `tools/binsnap_ancestry_guard.sh` REFUSED the binsnap off `1e8a06e`:
   `ANCESTRY MISSING f59aa35 p6z-lenient-metadata-and-reconciler-attribution`,
   `ANCESTRY REFUSED candidate=ff877e71… missing=1 of 16 declared fixes`. The guard is
   right — the declared fix set grew to 16 when B9b was re-cut.

**The original prediction stood and was met on its own base: 1744 / 0 / 21 off
`1e8a06e`, three consecutive full runs.** That number is not retracted; it is simply
about a base that is no longer the tip.

Re-derived for `1273d62`, whose own gate (`mB9b-gate` 5833379) predicted and printed
`EXPECT_PASS=1773`:

**PREDICTED SPLIT ON THE REBASED BRANCH: 1779 passed / 0 failed / 21 ignored** = 1773 + 6.

The rebase conflicted in three files against p6u/p6w's `AdvertisedIdentityRecord` work
(`auto_imports_suppressed_bundles`, `auto_imports_suppressed`,
`auto_imports_suppressed_roots`). Every hunk was resolved by keeping BOTH sides — no
p6u/p6w field or assertion is dropped — and `SCHEMA` becomes **6** (the new base is at
5, p6ad's own bump goes on top of it). No test is deleted, renamed or ignored, so the
1773 must be carried through unchanged.
