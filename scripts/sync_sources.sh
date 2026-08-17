#!/usr/bin/env bash
# Clone or update the two upstream data repos into vendor/.
#   vendor/MLB-PBP     buffedlizard55-lab/MLB-PBP      (scoreboard + full PBP)
#   vendor/StatcastMLB karagemop466-tech/StatcastMLB   (Statcast + leaderboards)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p vendor

sync() {
  local url="$1" dest="$2"
  if [ -d "$dest/.git" ]; then
    echo "updating $dest"
    git -C "$dest" pull --ff-only
  else
    echo "cloning $url -> $dest"
    git clone --depth 1 "$url" "$dest"
  fi
}

sync https://github.com/buffedlizard55-lab/MLB-PBP.git    vendor/MLB-PBP
sync https://github.com/karagemop466-tech/StatcastMLB.git vendor/StatcastMLB

echo
echo "Optional installs:"
echo "  pip install -e vendor/MLB-PBP       # stdlib-only PBP/scoreboard archiver"
echo "  pip install -e vendor/StatcastMLB   # Statcast harvesters + validators"
