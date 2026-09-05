#!/usr/bin/env bash
# p6ad mutation controls. A guard that cannot fail is a defect, so every guard
# below is shown RED under a mutation that reintroduces the bug it locks.
#
#   bash p6ad_negctl.sh            # baseline + every mutation
#   bash p6ad_negctl.sh baseline   # baseline only
#
# It edits a COPY of the worktree under $ARMS and never touches the branch.
set +u
WT=${WT:-/oscar/data/stellex/glvov/agrescap/worktrees/fix-p6ad}
ARMS=${ARMS:-/oscar/data/stellex/glvov/agrescap/tasks/retread-4-11/p6ad-work/arms}
export PATH="$HOME/.cargo/bin:$PATH"
mkdir -p "$ARMS"

run_arm () {                       # run_arm <name> <python mutation script or -->
  local name=$1; local mutate=$2; local dir="$ARMS/$name"
  rm -rf "$dir"; mkdir -p "$dir"
  # Sources only; the target dir is SHARED read-write with the baseline arm on
  # purpose (one build cache, many arms) via CARGO_TARGET_DIR.
  cp -a "$WT/src" "$WT/Cargo.toml" "$WT/Cargo.lock" "$WT/build.rs" \
        "$WT/rust-toolchain.toml" "$WT/recipe" "$WT/tests" "$WT/examples" "$dir/" 2>/dev/null
  if [ "$mutate" != "--" ]; then python3 "$mutate" "$dir/src" || { echo "$name: MUTATION DID NOT APPLY"; return 9; }; fi
  ( cd "$dir" && CARGO_TARGET_DIR="$ARMS/target" cargo test --lib p6ad 2>&1 | tail -25 ) > "$dir/out.txt" 2>&1
  echo "--- $name ---"
  grep -E '^test (repodata|handler)::tests::p6ad|^test result|^error' "$dir/out.txt" | tail -20
}

mut () { cat > "$ARMS/$1.py"; echo "$ARMS/$1.py"; }

# A: the memo believes the ADDRESS alone -- the stat tuple is dropped, so a
#    sidecar answers for whatever bytes are there now. This is the shape the
#    guard caught twice while p6ad was being built.
A=$(mut mutA <<'PY'
import sys,pathlib
p=pathlib.Path(sys.argv[1],'repodata.rs'); s=p.read_text()
old='        && value.get("stat").and_then(|v| v.as_str()) == Some(key.as_str())\n'
assert s.count(old)==1
p.write_text(s.replace(old,''))
PY
)
# B: the memo is consulted with NO freeze -- the fast path is live everywhere,
#    which is exactly the unsound configuration the guards forced out.
B=$(mut mutB <<'PY'
import sys,pathlib
p=pathlib.Path(sys.argv[1],'repodata.rs'); s=p.read_text()
old='    if frozen()\n        && let Some(sidecar) = sidecar.as_ref()'
assert s.count(old)==1
p.write_text(s.replace(old,'    if let Some(sidecar) = sidecar.as_ref()'))
PY
)
# C: the digest is a BAG OF HASHES -- the (channel, subdir) labels are dropped
#    from the fold, so two subdirs swapping documents look identical.
C=$(mut mutC <<'PY'
import sys,pathlib
p=pathlib.Path(sys.argv[1],'repodata.rs'); s=p.read_text()
old="""        hasher.update(document.channel.as_bytes());
        hasher.update([0x1fu8]);
        hasher.update(document.subdir.as_bytes());
        hasher.update([0x1fu8]);
        hasher.update(document.sha256.as_bytes());"""
assert s.count(old)==1
p.write_text(s.replace(old,"        hasher.update(document.sha256.as_bytes());"))
PY
)
# D: the fingerprint is LENGTH + MTIME -- the pre-v2 `repodata_identity` rule,
#    the one that discarded 13 of 14 verdict files when nothing had moved.
D=$(mut mutD <<'PY'
import sys,pathlib
p=pathlib.Path(sys.argv[1],'repodata.rs'); s=p.read_text()
old="    let (sha256, bytes) = hash_file_blocking(path)?;\n"
assert s.count(old)>=1
new=('    let (_ignored, bytes) = hash_file_blocking(path)?;\n'
     '    let sha256 = { use std::os::unix::fs::MetadataExt as _; format!("{:064x}", (meta.len() as u128) << 64 | meta.mtime() as u128) };\n')
p.write_text(s.replace(old,new,1))
PY
)
# E: the VERB labels on the slug alone -- two channel URLs that slug to one
#    name collide, which is what the live cache actually does with pytorch.
E=$(mut mutE <<'PY'
import sys,pathlib
p=pathlib.Path(sys.argv[1],'repodata.rs'); s=p.read_text()
old='            channel: format!("{}#{}", parts[0], parts[2]),'
assert s.count(old)==1
p.write_text(s.replace(old,'            channel: parts[0].to_string(),'))
PY
)

case "${1:-all}" in
  baseline) run_arm baseline -- ;;
  *) run_arm baseline --
     run_arm mutA "$A"; run_arm mutB "$B"; run_arm mutC "$C"; run_arm mutD "$D"; run_arm mutE "$E" ;;
esac
