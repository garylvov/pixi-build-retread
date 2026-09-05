#!/usr/bin/env bash
# p6af-2h PROBE 5 -- probe4's exact-line `strings | grep -x` returned 0 for EVERY
# env name including ones pixi certainly knows, so that reading is worthless on
# its own (the uv binary packs its strings into concatenated blobs -- we saw the
# whole Accept header glued to five other literals). Re-ask with SUBSTRING greps
# and a couple of known-present controls, because the answer decides whether the
# fallback design (a TLS-terminating proxy with a job-local CA) is even possible.
set -uo pipefail
# HARNESS-EXIT-2: the root is argv-overridable so `driver_exit_guard.sh` can run
# THIS FILE, unmodified, over a throwaway root. The default is the lane root and
# is unchanged, so every existing call site behaves exactly as before.
PIXIREAL=${1:-/oscar/data/stellex/glvov/homecache/pixi/bin/pixi.real}
echo "### P6AF2H PROBE5 start $(date -Is) host=$(hostname)"
ls -l "$PIXIREAL"
for n in SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE \
         NATIVE_TLS native-tls rustls webpki rustls-native-certs \
         UV_INDEX_URL PIXI_CACHE_DIR PIXI_HOME HTTPS_PROXY NO_PROXY \
         pypi-config index-url extra-index-urls mirrors; do
  c=$(strings -n 4 "$PIXIREAL" | grep -c -- "$n")
  echo "###   substring '$n' hits=$c"
done

# HARNESS-EXIT-2 (law 9). This probe used to end on an `echo`, so its exit
# status was the echo's and Slurm recorded 0:0 even when the probe never ran at
# all -- the same shape that let c181-dryrun (rc=8) and l3-2arm (job_fatal=1)
# report COMPLETED. The per-arm rcs above are DATA (arms pointed at a dead port
# are MEANT to fail and are not a job failure); what is fatal is the probe
# itself not producing the baseline artifact it exists to compare against.
# Every printed line above is unchanged.
PROBE_FATAL=0
[ -r "$PIXIREAL" ] || { echo "### PROBE5 FATAL: $PIXIREAL is not readable -- every hits=0 above is an artefact of that, not a reading"; PROBE_FATAL=1; }
echo "### P6AF2H PROBE5 DONE $(date -Is) probe_fatal=$PROBE_FATAL"
exit "$PROBE_FATAL"
