#!/usr/bin/env bash
# Point the (single) live channel at a new stream link and start transcribing it.
#   scripts/set_stream.sh udp://@239.1.1.1:5000        (or just 239.1.1.1:5000)
#   API=http://server:8000 STT_API_TOKEN=... scripts/set_stream.sh ...
# Uses the running API (no restart needed): the channel's worker reconnects within ~10 s.
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 <stream link> [channel id, default ch1]" >&2; exit 2; }
url="$1"; ch="${2:-ch1}"; api="${API:-http://localhost:8000}"
if [[ ! "$url" =~ ^[a-zA-Z][a-zA-Z0-9+.-]*:// ]]; then        # bare host:port → udp://
  first="${url#@}"; first="${first%%.*}"
  if [[ "$first" =~ ^[0-9]+$ && "$first" -ge 224 && "$first" -le 239 ]]; then
    url="udp://@${url#@}"      # multicast: join the group
  else
    url="udp://$url"
  fi
fi
auth=()
if [[ -n "${STT_API_TOKEN:-}" ]]; then auth=(-H "Authorization: Bearer $STT_API_TOKEN"); fi

call() {  # call METHOD PATH JSON → sets CODE and OUT
  local r
  r=$(curl -s -w $'\n%{http_code}' -X "$1" ${auth[@]+"${auth[@]}"} -H 'Content-Type: application/json' -d "$3" "$api$2")
  CODE="${r##*$'\n'}"
  OUT="${r%$'\n'*}"
}

call PATCH "/channels/$ch" "$(printf '{"url": "%s", "enabled": true}' "$url")"
if [[ "$CODE" == "404" ]]; then
  call POST "/channels" "$(printf '{"id": "%s", "name": "Channel 1", "url": "%s", "audio_track": {"index": 0, "language": "urd"}, "language": "ur", "enabled": true}' "$ch" "$url")"
fi
if [[ "$CODE" =~ ^20 ]]; then
  echo "OK: $ch now reads $url"
  echo "Open $api  (video and Urdu text start about 20 s after the stream arrives)"
else
  echo "FAILED ($CODE): $OUT" >&2
  exit 1
fi
