#!/usr/bin/env bash
# p6af-2h PROBE 5 -- probe4's exact-line `strings | grep -x` returned 0 for EVERY
# env name including ones pixi certainly knows, so that reading is worthless on
# its own (the uv binary packs its strings into concatenated blobs -- we saw the
# whole Accept header glued to five other literals). Re-ask with SUBSTRING greps
# and a couple of known-present controls, because the answer decides whether the
# fallback design (a TLS-terminating proxy with a job-local CA) is even possible.
set -uo pipefail
PIXIREAL=/oscar/data/stellex/glvov/homecache/pixi/bin/pixi.real
echo "### P6AF2H PROBE5 start $(date -Is) host=$(hostname)"
ls -l "$PIXIREAL"
for n in SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE \
         NATIVE_TLS native-tls rustls webpki rustls-native-certs \
         UV_INDEX_URL PIXI_CACHE_DIR PIXI_HOME HTTPS_PROXY NO_PROXY \
         pypi-config index-url extra-index-urls mirrors; do
  c=$(strings -n 4 "$PIXIREAL" | grep -c -- "$n")
  echo "###   substring '$n' hits=$c"
done
echo "### P6AF2H PROBE5 DONE $(date -Is)"
