#!/usr/bin/env bash
#
# Run the app under Apple's native container runtime (https://github.com/apple/container).
#
#   ./scripts/apple-container.sh build    build the three images
#   ./scripts/apple-container.sh up       build if needed, then start vision, video and the web app
#   ./scripts/apple-container.sh down     stop and remove the three containers (volumes are kept)
#   ./scripts/apple-container.sh ps       show the containers and where they listen
#   ./scripts/apple-container.sh logs [service]   follow a service's output (web by default)
#   ./scripts/apple-container.sh url      print the app's address
#
# Docker and Podman use compose.yaml instead. This script exists for three reasons:
#
#   * Apple container has no Compose support of its own. `container-compose` is a
#     third-party shim and silently drops CPU limits, ulimits and restart policies.
#   * `container` gives every container 4 CPUs and 1 GB of RAM unless told otherwise
#     (Docker's single VM gets the whole machine). The model services need more, and
#     the sizes below are what keep them at full speed.
#   * A named volume can only be attached to one container at a time here, and two
#     containers can't see each other by name, so /videos is a host folder shared by
#     bind mount and the web app is pointed at the other services by IP.
#
# Setup, once:
#
#   brew install container && container system start
#
# Reaching a model server on the Mac from inside a container needs a DNS domain that
# Apple container can resolve, which takes a sudo password once per machine:
#
#   sudo container system dns create host.container.internal --localhost 203.0.113.113
#
# Skip that and put your Mac's IP in .env instead (it changes with your network):
#
#   echo 'LLM_BASE_URL=http://192.168.1.10:8080/v1' > .env
#
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

# ---------------------------------------------------------------- configuration

# Read .env the way `docker compose` does: KEY=value lines, # comments ignored.
if [ -f .env ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line#"${line%%[![:space:]]*}"}   # trim leading space
    case $line in ''|'#'*) continue ;; esac
    key=${line%%=*}
    value=${line#*=}
    [ "$key" = "$line" ] && continue
    case $key in *[!A-Za-z0-9_]*|'') continue ;; esac
    value=${value%\"}; value=${value#\"}
    value=${value%\'}; value=${value#\'}
    export "$key=$value"
  done < .env
fi

: "${LLM_BASE_URL:=http://host.container.internal:8080/v1}"
: "${LLM_API_KEY:=}"
: "${LLM_TIMEOUT:=300}"
: "${VISION_MAX_LOADED:=2}"
: "${VIDEO_RETENTION_DAYS:=7}"
: "${APP_PORT:=8090}"
: "${VISION_PORT:=8000}"

# CPU and memory per container. Apple's defaults (4 CPUs, 1 GB) are too small for the
# model services; Docker's VM gets the whole machine, so the vision and video services
# are given every core here too and the two runtimes stay comparable.
NCPUS=$(sysctl -n hw.ncpu 2>/dev/null || echo 8)
: "${APP_CPUS:=2}"
: "${APP_MEMORY:=2g}"
: "${VISION_CPUS:=$NCPUS}"
: "${VISION_MEMORY:=8g}"
: "${VIDEO_CPUS:=$NCPUS}"
: "${VIDEO_MEMORY:=8g}"

# compose.yaml keeps /videos in the video-data volume, which the web app and the video
# service both mount. Apple container can only attach a named volume to one container at
# a time (the second fails with "The storage device attachment is invalid"), so /videos is
# a host folder both containers bind mount instead. A bind mount can be shared; a named
# volume cannot. Nothing but video uploads lives here.
: "${VIDEO_DATA:=$ROOT/video-data}"

WEB=llm-classifier
VISION=llm-classifier-vision
VIDEO=llm-classifier-video
VOL_APP=llm-classifier_app-data
VOL_VISION_CACHE=llm-classifier_vision-cache
VOL_VIDEO_CACHE=llm-classifier_video-cache

# ---------------------------------------------------------------- helpers

note() { printf '\033[1m==>\033[0m %s\n' "$*" >&2; }
die() { printf 'Error: %s\n' "$*" >&2; exit 1; }

need_container() {
  command -v container >/dev/null 2>&1 ||
    die "Apple's container CLI is not installed. Install it with: brew install container"
  command -v python3 >/dev/null 2>&1 ||
    die "python3 is needed to read container addresses. Install it with: xcode-select --install"
  case "$(uname -m)" in
    arm64) ;;
    *) die "container only runs on Apple silicon." ;;
  esac
}

ensure_system() {
  need_container
  if container system status 2>/dev/null | grep -qE '^status +running'; then
    return
  fi
  note "starting the container system service"
  # The first start asks whether to install the recommended kernel; answer yes.
  yes 2>/dev/null | container system start >/dev/null ||
    die "could not start the container system service"
  container system status 2>/dev/null | grep -qE '^status +running' ||
    die "the container system service did not come up"
}

ensure_builder() {
  # Builds run in a helper container that defaults to 2 CPUs and 2 GB. torch and
  # transformers unpack more than that on a 16-core machine, so start it wider.
  # `container builder status` prints "builder is not running" when it is down.
  if ! container builder status 2>&1 | grep -qi 'not running'; then
    return
  fi
  note "starting the image builder with $NCPUS CPUs and ${BUILDER_MEMORY:-8g} of memory"
  container builder start --cpus "$NCPUS" --memory "${BUILDER_MEMORY:-8g}" >/dev/null
}

build_image() {
  local tag=$1 dir=$2 file=$3
  note "building $tag"
  (cd "$dir" && container build --tag "$tag" --file "$file" .)
}

image_exists() { container image ls 2>/dev/null | awk '{print $1}' | grep -qx "$1"; }

ensure_volume() {
  container volume ls 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx "$1" ||
    container volume create "$1" >/dev/null
}

# Docker gives a fresh named volume the ownership of the image folder it covers, so the
# app's uid 1001 can write to /cache and /data. `container` starts every volume empty and
# root-owned, and model loading then fails with "PermissionError: /cache/hf", so hand the
# folder to the app's uid once, the first time the volume is used.
prepare_volume() {
  local volume=$1 image=$2 path=$3 owner
  owner=$(container run --rm --user 0 --volume "$volume:$path" "$image" \
    stat -c '%u:%g' "$path" 2>/dev/null | tr -d '\r' | tail -1)
  if [ "$owner" = "1001:1001" ]; then
    return 0
  fi
  note "handing $volume to the app's uid"
  container run --rm --user 0 --volume "$volume:$path" "$image" chown 1001:1001 "$path" >/dev/null
}

# Where a container's IPv4 address is, without the /24.
container_ip() {
  container inspect "$1" 2>/dev/null |
    python3 -c 'import json,sys
try:
    print(json.load(sys.stdin)[0]["status"]["networks"][0]["ipv4Address"].split("/")[0])
except (KeyError, IndexError, TypeError):
    pass'
}

wait_for_ip() {
  local name=$1 ip=""
  for _ in $(seq 1 60); do
    ip=$(container_ip "$name")
    [ -n "$ip" ] && { printf '%s' "$ip"; return 0; }
    sleep 1
  done
  return 1
}

wait_for_http() {
  local url=$1
  for _ in $(seq 1 90); do
    if curl -fsS -o /dev/null --max-time 2 "$url" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

replace_container() {
  local name=$1
  if container inspect "$name" >/dev/null 2>&1; then
    note "replacing the running $name"
    container stop "$name" >/dev/null 2>&1 || true
    container delete "$name" >/dev/null 2>&1 || true
  fi
}

# ---------------------------------------------------------------- commands

do_build() {
  ensure_system
  ensure_builder
  # vision and video share a base layer; building vision first lets video reuse it
  build_image "$WEB" "$ROOT" Containerfile
  build_image "$VISION" "$ROOT/vision" Containerfile
  build_image "$VIDEO" "$ROOT/video" Containerfile
  note "built $WEB, $VISION and $VIDEO"
}

do_up() {
  ensure_system
  local missing=0
  for image in "$WEB" "$VISION" "$VIDEO"; do
    if ! image_exists "$image"; then
      missing=1
    fi
  done
  if [ "$missing" = 1 ]; then
    do_build
  fi

  # Stop anything still running first: a volume can only be attached to one container,
  # so a container left over from a previous run would block its own volume.
  replace_container "$WEB"
  replace_container "$VISION"
  replace_container "$VIDEO"

  ensure_volume "$VOL_APP"
  ensure_volume "$VOL_VISION_CACHE"
  ensure_volume "$VOL_VIDEO_CACHE"
  prepare_volume "$VOL_APP" "$WEB" /data
  prepare_volume "$VOL_VISION_CACHE" "$VISION" /cache
  prepare_volume "$VOL_VIDEO_CACHE" "$VIDEO" /cache
  # Both the web app and the video service bind mount this one host folder.
  mkdir -p "$VIDEO_DATA"
  VIDEO_DATA=$(cd "$VIDEO_DATA" && pwd)

  note "starting $VISION"
  container run --detach --name "$VISION" \
    --cpus "$VISION_CPUS" --memory "$VISION_MEMORY" \
    --env "VISION_MAX_LOADED=$VISION_MAX_LOADED" \
    --volume "$VOL_VISION_CACHE:/cache" \
    --publish "127.0.0.1:$VISION_PORT:8000" \
    "$VISION" >/dev/null

  note "starting $VIDEO"
  container run --detach --name "$VIDEO" \
    --cpus "$VIDEO_CPUS" --memory "$VIDEO_MEMORY" \
    --env "VIDEO_RETENTION_DAYS=$VIDEO_RETENTION_DAYS" \
    --volume "$VIDEO_DATA:/videos" \
    --volume "$VOL_VIDEO_CACHE:/cache" \
    "$VIDEO" >/dev/null

  # Apple container does not resolve one container's name from another (apple/container#1809),
  # and it has no --add-host, so the web app is pointed at the two services by IP. Read them
  # after start; run `up` again after a reboot to pick up new ones.
  local vision_ip video_ip
  vision_ip=$(wait_for_ip "$VISION") || die "$VISION did not get an IP address"
  video_ip=$(wait_for_ip "$VIDEO") || die "$VIDEO did not get an IP address"

  note "starting $WEB (vision $vision_ip, video $video_ip)"
  container run --detach --name "$WEB" \
    --cpus "$APP_CPUS" --memory "$APP_MEMORY" \
    --env "LLM_BASE_URL=$LLM_BASE_URL" \
    --env "LLM_API_KEY=$LLM_API_KEY" \
    --env "LLM_TIMEOUT=$LLM_TIMEOUT" \
    --env "VISION_BASE_URL=http://$vision_ip:8000" \
    --env "VIDEO_BASE_URL=http://$video_ip:8001" \
    --volume "$VOL_APP:/data" \
    --volume "$VIDEO_DATA:/videos" \
    --publish "127.0.0.1:$APP_PORT:8080" \
    "$WEB" >/dev/null

  wait_for_http "http://127.0.0.1:$VISION_PORT/health" ||
    die "$VISION did not answer on port $VISION_PORT; run: ./scripts/apple-container.sh logs vision"
  wait_for_http "http://127.0.0.1:$APP_PORT/api/config" ||
    die "the web app did not answer on port $APP_PORT; run: ./scripts/apple-container.sh logs"

  note "app on http://127.0.0.1:$APP_PORT  ·  vision on 127.0.0.1:$VISION_PORT for the CLI"
  case $LLM_BASE_URL in
    *host.container.internal*)
      if container system dns list 2>/dev/null | grep -q host.container.internal; then
        note "host.container.internal is registered; llama-server on the Mac is reachable"
      else
        printf '\n' >&2
        printf 'Note: a llama-server on the Mac is not reachable yet. Register the domain once:\n' >&2
        printf '  sudo container system dns create host.container.internal --localhost 203.0.113.113\n' >&2
        printf 'or put your Mac'"'"'s IP in .env: LLM_BASE_URL=http://<mac-ip>:8080/v1\n' >&2
      fi
      ;;
  esac
}

do_down() {
  ensure_system
  for name in "$WEB" "$VISION" "$VIDEO"; do
    if container inspect "$name" >/dev/null 2>&1; then
      note "stopping $name"
      container stop "$name" >/dev/null 2>&1 || true
      container delete "$name" >/dev/null 2>&1 || true
    fi
  done
  note "stopped. Volumes kept; delete them with: container volume delete $VOL_APP"
}

do_ps() {
  ensure_system
  container list --all
}

do_logs() {
  ensure_system
  local name="$WEB"
  case ${1:-} in
    web) name=$WEB ;;
    vision) name=$VISION ;;
    video) name=$VIDEO ;;
    '') ;;
    *) die "unknown service '$1'. Use web, vision or video." ;;
  esac
  container logs --follow "$name"
}

do_url() {
  printf 'http://127.0.0.1:%s\n' "$APP_PORT"
}

usage() {
  # The comment block at the top of this file is the help text
  awk 'NR>2 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "$0"
}

case ${1:-} in
  build) shift; do_build "$@" ;;
  up) shift; do_up "$@" ;;
  down) shift; do_down "$@" ;;
  ps) shift; do_ps "$@" ;;
  logs) shift; do_logs "$@" ;;
  url) shift; do_url "$@" ;;
  -h|--help|help|'') usage ;;
  *) printf 'Unknown command: %s\n\n' "$1" >&2; usage >&2; exit 1 ;;
esac