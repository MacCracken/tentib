#!/usr/bin/env bash
# bench-history.sh — run tentib's benchmark harness and append the results to
# bench-history.csv, one row per benchmark, stamped with the date, VERSION and
# commit. It also prints this run as a Markdown table in human-readable units.
#
# This is the baseline tool of the first-party P(-1) process: run it before a
# hardening or optimization pass and again after, and compare the two stamps.
# Adapted from the tyche / rosnet / akshara scripts of the same name. tentib's
# harness prints its own lines rather than lib/bench.cyr's report, so the parser
# differs.
#
# The harness is built ONCE (CYRIUS_DCE=1, as CI builds it) and that one binary
# is run RUNS times (default 3), pinned to CPU $BENCH_CPU (default 4) when
# taskset exists. docs/benchmarks.md says why: an idle physical core, because a
# busy SMT sibling inflates every number. Each recorded value is the median of
# the runs. Compare rows only when they were recorded in the same session on the
# same host.
#
# Another machine (1.6.x, the edge target): BENCH_HOST=<ssh host> builds the
# harness for aarch64 here, copies it to a temporary file on that host, runs it
# there RUNS times (pinned to BENCH_CPU, default the host's last CPU, when the
# host has taskset), and removes it. Every row is recorded as "<label>: <row>",
# the label being BENCH_LABEL (default: the host name), so the rows of different
# machines never mix. The host must be aarch64 Linux and reachable without a
# password prompt (ssh BatchMode); the script prints its model, clock governor
# and temperature before and after the runs, for the record.
#   BENCH_HOST=pi ./scripts/bench-history.sh
#
# Parsed line shapes (tests/tentib.bcyr):
#   "128x128  branchy=105726  branchless=85944  simd=1663  rosnet-f64=6646  ns/call ..."
#   "  tsimd_pack_w=4846511  act_quant_int=6632"            (after the [2] header)
#   "  f64 weight-only        = 10382624 ns/fwd  (3082 tok/s)"
#   "  ... (one-time tx_pack = 8261595 ns, amortized over the serve)"
#   "  training step (f64)    = 1234567 ns/step"
#   "  serialize              = 123456 ns/op  (339560 bytes)"
#   "  decode, cached step    = 123456 ns/tok  (...)"           (1.4.0)
set -euo pipefail

cd "$(dirname "$0")/.."

CSV="bench-history.csv"
BENCH="tests/tentib.bcyr"
RUNS="${RUNS:-3}"
HOST="${BENCH_HOST:-}"
LABEL="${BENCH_LABEL:-$HOST}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)
if [ -n "$HOST" ]; then
    # The host name goes to ssh as an argument: a leading '-' would be read as an option.
    case "$HOST" in
        -*|*[!A-Za-z0-9._@-]*) echo "BENCH_HOST must be a plain ssh host name, got '$HOST'" >&2; exit 2 ;;
    esac
    case "$LABEL" in
        ''|*[!A-Za-z0-9._@\ -]*) echo "BENCH_LABEL must be letters, digits, space, '.', '_', '@' or '-', got '$LABEL'" >&2; exit 2 ;;
    esac
    ARCH="$("${SSH[@]}" "$HOST" uname -m)" || { echo "cannot reach $HOST over ssh" >&2; exit 1; }
    [ "$ARCH" = "aarch64" ] || { echo "$HOST is $ARCH; only aarch64 hosts are supported" >&2; exit 2; }
    CPU="${BENCH_CPU:-$(( $("${SSH[@]}" "$HOST" nproc) - 1 ))}"
else
    CPU="${BENCH_CPU:-4}"
fi
case "$RUNS" in
    ''|*[!0-9]*) echo "RUNS must be a positive integer, got '$RUNS'" >&2; exit 2 ;;
esac
[ "$RUNS" -ge 1 ] || { echo "RUNS must be at least 1" >&2; exit 2; }
case "$CPU" in
    ''|*[!0-9]*) echo "BENCH_CPU must be a CPU number, got '$CPU'" >&2; exit 2 ;;
esac

DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
VERSION="$(tr -d '[:space:]' < VERSION)"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    COMMIT="${COMMIT}+dirty"
fi

if [ ! -f "$CSV" ]; then
    echo "date,version,commit,benchmark,time_ns" > "$CSV"
fi

TMPDIR_B="$(mktemp -d)"
REMOTE=""
cleanup() {
    rm -rf "$TMPDIR_B"
    if [ -n "$REMOTE" ]; then "${SSH[@]}" "$HOST" rm -f "$REMOTE" || true; fi
}
trap cleanup EXIT

TARGET=()
if [ -n "$HOST" ]; then TARGET=(--aarch64); fi
echo "Building: CYRIUS_DCE=1 cyrius build ${TARGET[*]:+${TARGET[*]} }$BENCH"
CYRIUS_DCE=1 cyrius build "${TARGET[@]}" "$BENCH" "$TMPDIR_B/bench" > "$TMPDIR_B/build.log" 2>&1 \
    || { cat "$TMPDIR_B/build.log" >&2; echo "build failed" >&2; exit 1; }

# The host's state, before and after the runs: a throttled or busy board reads slow.
host_state() {
    "${SSH[@]}" "$HOST" 'printf "%s; " "$(tr -d "\0" < /proc/device-tree/model 2>/dev/null || uname -m)";
        printf "governor %s, " "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)";
        printf "%s MHz now, " "$(( $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo 0) / 1000 ))";
        printf "%s C, " "$(( $(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0) / 1000 ))";
        printf "load %s\n" "$(cut -d " " -f 1-3 /proc/loadavg)"'
}

PIN=()
if [ -n "$HOST" ]; then
    REMOTE="$("${SSH[@]}" "$HOST" mktemp /tmp/tentib-bench.XXXXXX)"
    scp -q -o BatchMode=yes "$TMPDIR_B/bench" "$HOST:$REMOTE"
    "${SSH[@]}" "$HOST" chmod 700 "$REMOTE"
    if "${SSH[@]}" "$HOST" command -v taskset > /dev/null 2>&1; then PIN=(taskset -c "$CPU"); fi
    echo "Host $HOST ($LABEL): $(host_state)"
elif command -v taskset > /dev/null 2>&1; then
    PIN=(taskset -c "$CPU")
fi

# One "<label>\t<ns>" line per benchmark per run.
i=1
while [ "$i" -le "$RUNS" ]; do
    if [ -n "$HOST" ]; then
        echo "Run $i of $RUNS: $HOST: ${PIN[*]:-} $REMOTE"
        "${SSH[@]}" "$HOST" "${PIN[@]}" "$REMOTE" > "$TMPDIR_B/out" 2>&1 \
            || { cat "$TMPDIR_B/out" >&2; echo "benchmark run failed on $HOST" >&2; exit 1; }
    else
        echo "Run $i of $RUNS: ${PIN[*]:-} $TMPDIR_B/bench"
        "${PIN[@]}" "$TMPDIR_B/bench" > "$TMPDIR_B/out" 2>&1 \
            || { cat "$TMPDIR_B/out" >&2; echo "benchmark run failed" >&2; exit 1; }
    fi
    awk '
        /^[0-9]+x[0-9]+ / {
            shape = $1
            for (f = 2; f <= NF; f++) {
                if ($f == "ns/call") break
                if (split($f, kv, "=") == 2 && kv[2] ~ /^[0-9]+$/) printf "layer %s %s\t%s\n", shape, kv[1], kv[2]
            }
            next
        }
        /^\[2\]/ { prep = 1; next }
        /^\[/ { prep = 0 }
        prep && /^[[:space:]]+[a-z_]+=[0-9]+/ {
            for (f = 1; f <= NF; f++)
                if (split($f, kv, "=") == 2 && kv[2] ~ /^[0-9]+$/) printf "prep 768x768 %s\t%s\n", kv[1], kv[2]
            next
        }
        / = [0-9]+ ns\/(fwd|step|op|tok)/ {
            line = $0
            sub(/^[[:space:]]+/, "", line)
            name = line; sub(/[[:space:]]*= [0-9]+ ns\/.*/, "", name)
            v = line; sub(/^[^=]*= /, "", v); sub(/ .*/, "", v)
            printf "model %s\t%s\n", name, v
            if (match(line, /one-time tx_pack = [0-9]+ ns/)) {
                t = substr(line, RSTART, RLENGTH); gsub(/[^0-9]/, "", t)
                printf "model tx_pack (one-time)\t%s\n", t
            }
        }
    ' "$TMPDIR_B/out" >> "$TMPDIR_B/rows"
    i=$((i + 1))
done

[ -s "$TMPDIR_B/rows" ] || { echo "no benchmark results parsed" >&2; exit 1; }
if [ -n "$HOST" ]; then echo "Host $HOST after the runs: $(host_state)"; fi

# Median per benchmark, in first-seen order. With an even RUNS it takes the
# lower middle value, so every recorded number is one that was measured.
awk -F '\t' '
    { if (!($1 in n)) order[++k] = $1; v[$1, ++n[$1]] = $2 }
    END {
        for (j = 1; j <= k; j++) {
            name = order[j]; m = n[name]
            for (a = 1; a <= m; a++) x[a] = v[name, a]
            for (a = 2; a <= m; a++) { t = x[a]; b = a - 1
                while (b >= 1 && x[b] > t) { x[b + 1] = x[b]; b-- } x[b + 1] = t }
            printf "%s\t%d\n", name, x[int((m + 1) / 2)]
        }
    }' "$TMPDIR_B/rows" > "$TMPDIR_B/median"

PREFIX=""
if [ -n "$HOST" ]; then PREFIX="$LABEL: "; fi
awk -F '\t' -v date="$DATE" -v ver="$VERSION" -v c="$COMMIT" -v pre="$PREFIX" '
    { name = pre $1; gsub(/"/, "\"\"", name); printf "%s,%s,%s,\"%s\",%s\n", date, ver, c, name, $2 }
' "$TMPDIR_B/median" >> "$CSV"

echo
echo "| Benchmark | Time (median of $RUNS) |"
echo "|---|---:|"
awk -F '\t' '{
    ns = $2 + 0
    if (ns >= 1e9) s = sprintf("%.2f s", ns / 1e9)
    else if (ns >= 1e6) s = sprintf("%.2f ms", ns / 1e6)
    else if (ns >= 1e3) s = sprintf("%.2f us", ns / 1e3)
    else s = sprintf("%d ns", ns)
    printf "| %s | %s |\n", $1, s
}' "$TMPDIR_B/median"
echo
echo "Appended $(wc -l < "$TMPDIR_B/median") rows to $CSV ($VERSION, $COMMIT, median of $RUNS runs${HOST:+, on $HOST})"
