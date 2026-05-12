#!/bin/bash

# Claude Agent Coordination Helper Scripts
# These commands help you view and interact with multi-agent coordination

echo "🤖 Claude Multi-Agent Coordination Commands:"
echo
echo "📋 Activity & Status:"
echo "   claude-agents-log            Recent activity"
echo "   claude-agents-active         Agents active in last 24h"
echo "   claude-agents-today          Today's events"
echo "   claude-agents-yesterday      Yesterday's events"
echo "   claude-agents-search <kw>    Search events"
echo "   claude-agents-stats          Statistics"
echo "   claude-agents-status         Overview of agents + tasks"
echo "   claude-agents-monitor        Live activity (Ctrl-C to stop)"
echo "   claude-agents-analyze        Quick repository analysis"
echo
echo "📝 Tasks:"
echo "   claude-agents-tasks          List tasks"
echo "   claude-agents-task-add <id> <description> [deadline-iso]"
echo "   claude-agents-assign <id> [deadline-iso]   Take ownership"
echo "   claude-agents-task-done <id>"
echo
echo "💬 Messaging:"
echo "   claude-agents-broadcast <message>"
echo "   claude-agents-dm <agent-id> <message>"
echo "   claude-agents-inbox          Messages addressed to you"
echo
echo "🔒 Locks:"
echo "   claude-agents-locks                     List active advisory locks"
echo "   claude-agents-locks-clear [stale|mine|all]   Clear locks (default stale)"
echo
echo "🧹 Maintenance:"
echo "   claude-agents-cleanup"
echo

# ---------- internal helpers ----------

_claude_coord_dir() {
    # Locate the repo root and resolve .claude/coordination
    local root
    root=$(git rev-parse --show-toplevel 2>/dev/null) || root="$PWD"
    echo "$root/.claude/coordination"
}

_claude_coord_init() {
    local dir
    dir=$(_claude_coord_dir)
    mkdir -p "$dir/locks" 2>/dev/null
    [ -f "$dir/tasks.json" ]    || echo '{"tasks": []}' > "$dir/tasks.json"
    [ -f "$dir/messages.log" ]  || : > "$dir/messages.log"
}

_claude_agent_id() {
    echo "${CLAUDE_AGENT_ID:-claude-${USER:-agent}-${HOSTNAME:-local}${CLAUDE_SESSION_ID:+-$CLAUDE_SESSION_ID}}"
}

_claude_log_event() {
    # Append a structured line to git notes (best-effort).
    local kind=$1; shift
    local payload="$*"
    local ts
    ts=$(date -Iseconds)
    local agent
    agent=$(_claude_agent_id)
    git notes --ref=agent-coordination append -m "$ts: $kind | agent: $agent | $payload" HEAD 2>/dev/null || true
}

# ---------- existing commands ----------

claude-agents-log() {
    echo "📜 Recent Agent Activity (Git Notes + Commits):"
    echo "═══════════════════════════════════════════════"
    echo "🗒️  Git Notes (agent-coordination):"
    git notes --ref=agent-coordination show HEAD 2>/dev/null | tail -20 || echo "   No coordination notes yet"
    echo
    echo "💬 Recent Agent Event Commits:"
    git log --oneline -20 --grep="AGENT_EVENT:" --all --format="%C(yellow)%h%C(reset) %C(green)%ad%C(reset) %s" --date=relative || echo "   No agent event commits yet"
}

claude-agents-active() {
    echo "👥 Active Agents (last 24 hours):"
    echo "═══════════════════════════════════════════"
    {
        git log --since="24 hours ago" --grep="AGENT_EVENT:" --all --format="%s" 2>/dev/null | grep -o "agent: [^|]*" | sort -u
        git notes --ref=agent-coordination show HEAD 2>/dev/null | grep -E "$(date -d 'yesterday' '+%Y-%m-%d')|$(date '+%Y-%m-%d')" | grep -o "agent: [^|]*" | sort -u
    } | sort -u || echo "   No recent agent activity"
}

claude-agents-today() {
    echo "📅 Today's Agent Activity:"
    echo "════════════════════════════"
    local today
    today=$(date '+%Y-%m-%d')
    {
        git log --since="today" --grep="AGENT_EVENT:" --all --format="%C(blue)%ad%C(reset) | %s" --date=format:"%H:%M"
        git notes --ref=agent-coordination show HEAD 2>/dev/null | grep "$today" | sed "s/^/📝 /"
    } | sort -k1 2>/dev/null || echo "   No activity today"
}

claude-agents-yesterday() {
    echo "📅 Yesterday's Agent Activity:"
    echo "═════════════════════════════"
    local y
    y=$(date -d 'yesterday' '+%Y-%m-%d')
    {
        git log --since="yesterday" --until="today" --grep="AGENT_EVENT:" --all --format="%C(blue)%ad%C(reset) | %s" --date=format:"%H:%M"
        git notes --ref=agent-coordination show HEAD 2>/dev/null | grep "$y" | sed "s/^/📝 /"
    } | sort -k1 2>/dev/null || echo "   No activity yesterday"
}

claude-agents-search() {
    if [ -z "$1" ]; then
        echo "Usage: claude-agents-search <keyword>"
        return 1
    fi
    echo "🔍 Searching agent events for: '$1'"
    echo "════════════════════════════════════════"
    {
        git log --grep="AGENT_EVENT:" --all --format="%C(yellow)%h%C(reset) %C(green)%ad%C(reset) %s" --date=relative | grep -i "$1"
        git notes --ref=agent-coordination show HEAD 2>/dev/null | grep -i "$1" | sed "s/^/📝 /"
    } | head -20 || echo "   No matches found for '$1'"
}

claude-agents-stats() {
    echo "📊 Agent Coordination Statistics:"
    echo "════════════════════════════════"
    echo "📈 Event Types (last 50 events):"
    {
        git log -50 --grep="AGENT_EVENT:" --all --format="%s" 2>/dev/null | grep -oE "AGENT_EVENT: [a-z_]+" | sed 's/AGENT_EVENT: //' | sort | uniq -c | sort -nr
        git notes --ref=agent-coordination show HEAD 2>/dev/null | tail -50 | grep -oE ": [A-Z_]+ \|" | sed -E 's/^: //; s/ \|$//' | sort | uniq -c | sort -nr
    } | head -10
    echo
    echo "🤖 Most Active Agents:"
    {
        git log -50 --grep="AGENT_EVENT:" --all --format="%s" 2>/dev/null | grep -oE "agent: [^|]+" | sed 's/agent: //; s/ *$//' | sort | uniq -c | sort -nr
        git notes --ref=agent-coordination show HEAD 2>/dev/null | tail -50 | grep -oE "agent: [^|]+" | sed 's/agent: //; s/ *$//' | sort | uniq -c | sort -nr
    } | head -5
}

claude-agents-cleanup() {
    echo "🧹 Cleaning up old agent coordination data..."
    git log --since="30 days ago" --until="now" --grep="AGENT_EVENT:" --format="%H" --all | head -100 > /tmp/keep_commits.txt
    git notes --ref=agent-coordination show HEAD 2>/dev/null | tail -1000 > /tmp/coordination_notes_clean.txt
    if [ -s /tmp/coordination_notes_clean.txt ]; then
        git notes --ref=agent-coordination remove HEAD 2>/dev/null || true
        cat /tmp/coordination_notes_clean.txt | git notes --ref=agent-coordination add HEAD
    fi
    rm -f /tmp/keep_commits.txt /tmp/coordination_notes_clean.txt
    echo "✅ Cleanup complete"
}

# ---------- new: status / monitor / analyze ----------

claude-agents-status() {
    _claude_coord_init
    echo "🤖 Agent Coordination Status"
    echo "════════════════════════════"
    echo "You: $(_claude_agent_id)"
    echo
    echo "Active agents (last 24h):"
    claude-agents-active | tail -n +3
    echo
    echo "Open tasks:"
    claude-agents-tasks open
}

claude-agents-monitor() {
    echo "🔭 Live coordination monitor (Ctrl-C to exit)"
    local last=""
    while true; do
        local now
        now=$(git notes --ref=agent-coordination show HEAD 2>/dev/null | tail -5)
        if [ "$now" != "$last" ]; then
            echo "── $(date '+%H:%M:%S') ──"
            echo "$now"
            last="$now"
        fi
        sleep 5
    done
}

claude-agents-analyze() {
    local root
    root=$(git rev-parse --show-toplevel 2>/dev/null) || root="$PWD"
    echo "🔍 Repository: $(basename "$root")"
    echo "Path: $root"
    if [ -f "$root/package.json" ]; then echo "Type: node"
    elif [ -f "$root/pyproject.toml" ] || [ -f "$root/requirements.txt" ]; then echo "Type: python"
    elif [ -f "$root/go.mod" ]; then echo "Type: go"
    elif [ -f "$root/Cargo.toml" ]; then echo "Type: rust"
    else echo "Type: generic"
    fi
    echo "Branch: $(git -C "$root" branch --show-current 2>/dev/null)"
    echo "Tracked files: $(git -C "$root" ls-files 2>/dev/null | wc -l)"
    if [ -f "$root/README.md" ]; then
        echo
        echo "README excerpt:"
        head -5 "$root/README.md" | sed 's/^/  /'
    fi
}

# ---------- new: tasks ----------

claude-agents-tasks() {
    _claude_coord_init
    local filter=${1:-all}   # all | open | done | mine
    local file
    file=$(_claude_coord_dir)/tasks.json
    if ! command -v jq >/dev/null 2>&1; then
        echo "jq required for task commands" >&2; return 1
    fi
    local me
    me=$(_claude_agent_id)
    echo "📝 Tasks ($filter):"
    case "$filter" in
        open) jq -r '.tasks[] | select(.status=="open") | "  [\(.id)] \(.description) — assignee: \(.assignee // "—") deadline: \(.deadline // "—")"' "$file" ;;
        done) jq -r '.tasks[] | select(.status=="done") | "  [\(.id)] \(.description) — done by \(.assignee // "—")"' "$file" ;;
        mine) jq -r --arg me "$me" '.tasks[] | select(.assignee==$me) | "  [\(.id)] \(.description) — \(.status)"' "$file" ;;
        all|*) jq -r '.tasks[] | "  [\(.id)] (\(.status)) \(.description) — assignee: \(.assignee // "—")"' "$file" ;;
    esac
}

claude-agents-task-add() {
    if [ $# -lt 2 ]; then
        echo "Usage: claude-agents-task-add <id> <description> [deadline-iso]"; return 1
    fi
    _claude_coord_init
    local id=$1 desc=$2 deadline=${3:-}
    local file
    file=$(_claude_coord_dir)/tasks.json
    if jq -e --arg id "$id" '.tasks[] | select(.id==$id)' "$file" >/dev/null; then
        echo "❌ Task $id already exists"; return 1
    fi
    local tmp
    tmp=$(mktemp)
    jq --arg id "$id" --arg desc "$desc" --arg deadline "$deadline" --arg ts "$(date -Iseconds)" \
        '.tasks += [{id:$id, description:$desc, deadline:$deadline, status:"open", assignee:null, created_at:$ts}]' \
        "$file" > "$tmp" && mv "$tmp" "$file"
    _claude_log_event TASK_ADDED "task: $id | description: $desc"
    echo "✅ Task $id added"
}

claude-agents-assign() {
    if [ -z "$1" ]; then
        echo "Usage: claude-agents-assign <task-id> [deadline-iso]"; return 1
    fi
    _claude_coord_init
    local id=$1 deadline=${2:-}
    local file
    file=$(_claude_coord_dir)/tasks.json
    local me
    me=$(_claude_agent_id)
    if ! jq -e --arg id "$id" '.tasks[] | select(.id==$id)' "$file" >/dev/null; then
        echo "❌ Task $id not found (use claude-agents-task-add first)"; return 1
    fi
    local tmp
    tmp=$(mktemp)
    jq --arg id "$id" --arg me "$me" --arg deadline "$deadline" \
        '(.tasks[] | select(.id==$id)) |= (.assignee=$me | (if $deadline=="" then . else .deadline=$deadline end))' \
        "$file" > "$tmp" && mv "$tmp" "$file"
    _claude_log_event TASK_ASSIGNED "task: $id | deadline: ${deadline:-none}"
    echo "✅ $me assigned to $id"
}

claude-agents-task-done() {
    if [ -z "$1" ]; then echo "Usage: claude-agents-task-done <task-id>"; return 1; fi
    _claude_coord_init
    local id=$1
    local file
    file=$(_claude_coord_dir)/tasks.json
    local tmp
    tmp=$(mktemp)
    jq --arg id "$id" --arg ts "$(date -Iseconds)" \
        '(.tasks[] | select(.id==$id)) |= (.status="done" | .completed_at=$ts)' \
        "$file" > "$tmp" && mv "$tmp" "$file"
    _claude_log_event TASK_DONE "task: $id"
    echo "✅ Task $id marked done"
}

# ---------- new: messaging ----------

claude-agents-broadcast() {
    if [ -z "$1" ]; then echo "Usage: claude-agents-broadcast <message>"; return 1; fi
    _claude_coord_init
    local me
    me=$(_claude_agent_id)
    local ts
    ts=$(date -Iseconds)
    local msg="$*"
    echo "$ts | BROADCAST | from: $me | $msg" >> "$(_claude_coord_dir)/messages.log"
    _claude_log_event BROADCAST "$msg"
    echo "📣 broadcast sent"
}

claude-agents-dm() {
    if [ $# -lt 2 ]; then echo "Usage: claude-agents-dm <agent-id> <message>"; return 1; fi
    _claude_coord_init
    local to=$1; shift
    local msg="$*"
    local me
    me=$(_claude_agent_id)
    local ts
    ts=$(date -Iseconds)
    echo "$ts | DM | from: $me | to: $to | $msg" >> "$(_claude_coord_dir)/messages.log"
    _claude_log_event DM "to: $to | $msg"
    echo "✉️  dm to $to sent"
}

claude-agents-inbox() {
    _claude_coord_init
    local me
    me=$(_claude_agent_id)
    local file
    file=$(_claude_coord_dir)/messages.log
    echo "📥 Messages for $me (broadcasts + DMs to you):"
    grep -E "BROADCAST|to: $me " "$file" 2>/dev/null | tail -50 || echo "   (empty)"
}

# ---------- locks ----------

claude-agents-locks() {
    _claude_coord_init
    local dir
    dir=$(_claude_coord_dir)/locks
    mkdir -p "$dir" 2>/dev/null
    local now ttl
    now=$(date +%s)
    ttl=${CLAUDE_LOCK_TTL:-600}
    echo "🔒 Active advisory locks (TTL ${ttl}s):"
    local found=0
    local lock
    for lock in "$dir"/*.lock; do
        [ -f "$lock" ] || continue
        local agent path ts_epoch age stale
        agent=$(grep '^agent=' "$lock" | cut -d= -f2-)
        path=$(grep '^path=' "$lock" | cut -d= -f2-)
        ts_epoch=$(grep '^ts_epoch=' "$lock" | cut -d= -f2-)
        age=0
        [ -n "$ts_epoch" ] && age=$((now - ts_epoch))
        stale=""
        [ "$age" -ge "$ttl" ] && stale=" (stale)"
        printf "  %s\n    agent=%s  age=%ss%s\n" "$path" "$agent" "$age" "$stale"
        found=1
    done
    [ $found -eq 0 ] && echo "  (none)"
}

claude-agents-locks-clear() {
    _claude_coord_init
    local dir
    dir=$(_claude_coord_dir)/locks
    mkdir -p "$dir" 2>/dev/null
    local mode=${1:-stale}
    local now ttl me
    now=$(date +%s)
    ttl=${CLAUDE_LOCK_TTL:-600}
    me=$(_claude_agent_id)
    local removed=0 lock
    for lock in "$dir"/*.lock; do
        [ -f "$lock" ] || continue
        local agent ts_epoch age
        agent=$(grep '^agent=' "$lock" | cut -d= -f2-)
        ts_epoch=$(grep '^ts_epoch=' "$lock" | cut -d= -f2-)
        age=0; [ -n "$ts_epoch" ] && age=$((now - ts_epoch))
        case "$mode" in
            all)   rm -f "$lock"; removed=$((removed+1)) ;;
            mine)  [ "$agent" = "$me" ] && rm -f "$lock" && removed=$((removed+1)) ;;
            stale|*) [ "$age" -ge "$ttl" ] && rm -f "$lock" && removed=$((removed+1)) ;;
        esac
    done
    echo "🧹 cleared $removed lock(s) (mode=$mode)"
}

# Make functions available in current shell
export -f claude-agents-log claude-agents-active claude-agents-today claude-agents-yesterday
export -f claude-agents-search claude-agents-stats claude-agents-cleanup
export -f claude-agents-status claude-agents-monitor claude-agents-analyze
export -f claude-agents-tasks claude-agents-task-add claude-agents-assign claude-agents-task-done
export -f claude-agents-broadcast claude-agents-dm claude-agents-inbox
export -f claude-agents-locks claude-agents-locks-clear
