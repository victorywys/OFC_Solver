#!/usr/bin/env fish

# Claude Agent Coordination Helper Scripts for Fish Shell
# These commands help you view and interact with multi-agent coordination

echo "🤖 Claude Multi-Agent Coordination Commands (Fish Shell):"
echo
echo "📋 Activity & Status:"
echo "   claude-agents-log / -active / -today / -yesterday / -search <kw> / -stats"
echo "   claude-agents-status / -monitor / -analyze"
echo "📝 Tasks:"
echo "   claude-agents-tasks [open|done|mine|all]"
echo "   claude-agents-task-add <id> <description> [deadline-iso]"
echo "   claude-agents-assign <id> [deadline-iso]"
echo "   claude-agents-task-done <id>"
echo "💬 Messaging:"
echo "   claude-agents-broadcast <message>"
echo "   claude-agents-dm <agent-id> <message>"
echo "   claude-agents-inbox"
echo "🔒 Locks:"
echo "   claude-agents-locks / -clear [stale|mine|all]"
echo "🧹 Maintenance:"
echo "   claude-agents-cleanup"
echo

# Function definitions for Fish shell
function claude-agents-log
    echo "📜 Recent Agent Activity (Git Notes + Commits):"
    echo "═══════════════════════════════════════════════"

    # Show git notes first
    echo "🗒️  Git Notes (agent-coordination):"
    if git notes --ref=agent-coordination show HEAD 2>/dev/null | tail -20
        # Notes found
    else
        echo "   No coordination notes yet"
    end

    echo
    echo "💬 Recent Agent Event Commits:"
    if git log --oneline -20 --grep="AGENT_EVENT:" --all --format="%C(yellow)%h%C(reset) %C(green)%ad%C(reset) %s" --date=relative 2>/dev/null
        # Commits found
    else
        echo "   No agent event commits yet"
    end
end

function claude-agents-active
    echo "👥 Active Agents (last 24 hours):"
    echo "═══════════════════════════════════════════"

    # Get unique agents from last 24 hours
    set temp_file (mktemp)

    # From git log
    git log --since="24 hours ago" --grep="AGENT_EVENT:" --all --format="%s" 2>/dev/null | grep -o "agent: [^|]*" | sort | uniq >> $temp_file

    # From git notes - Fish compatible date handling
    set yesterday (date -d yesterday '+%Y-%m-%d' 2>/dev/null; or date -v-1d '+%Y-%m-%d' 2>/dev/null; or echo (date '+%Y-%m-%d'))
    set today (date '+%Y-%m-%d')
    git notes --ref=agent-coordination show HEAD 2>/dev/null | grep -E "$yesterday|$today" | grep -o "agent: [^|]*" | sort | uniq >> $temp_file

    if test -s $temp_file
        sort $temp_file | uniq
    else
        echo "   No recent agent activity"
    end

    rm -f $temp_file
end

function claude-agents-today
    echo "📅 Today's Agent Activity:"
    echo "════════════════════════════"
    set today (date '+%Y-%m-%d')
    set temp_file (mktemp)

    # Get commits from today
    git log --since="today" --grep="AGENT_EVENT:" --all --format="%C(blue)%ad%C(reset) | %s" --date=format:"%H:%M" 2>/dev/null >> $temp_file

    # Get notes from today
    git notes --ref=agent-coordination show HEAD 2>/dev/null | grep "$today" | sed "s/^/📝 /" >> $temp_file

    if test -s $temp_file
        sort $temp_file
    else
        echo "   No activity today"
    end

    rm -f $temp_file
end

function claude-agents-yesterday
    echo "📅 Yesterday's Agent Activity:"
    echo "═════════════════════════════"
    set yesterday (date -d yesterday '+%Y-%m-%d' 2>/dev/null; or date -v-1d '+%Y-%m-%d' 2>/dev/null; or echo (date '+%Y-%m-%d'))
    set temp_file (mktemp)

    # Get commits from yesterday
    git log --since="yesterday" --until="today" --grep="AGENT_EVENT:" --all --format="%C(blue)%ad%C(reset) | %s" --date=format:"%H:%M" 2>/dev/null >> $temp_file

    # Get notes from yesterday
    git notes --ref=agent-coordination show HEAD 2>/dev/null | grep "$yesterday" | sed "s/^/📝 /" >> $temp_file

    if test -s $temp_file
        sort $temp_file
    else
        echo "   No activity yesterday"
    end

    rm -f $temp_file
end

function claude-agents-search
    if test (count $argv) -eq 0
        echo "Usage: claude-agents-search <keyword>"
        echo "Example: claude-agents-search \"commit_complete\""
        return 1
    end

    set keyword $argv[1]
    echo "🔍 Searching agent events for: '$keyword'"
    echo "════════════════════════════════════════"

    set temp_file (mktemp)

    # Search in git log
    git log --grep="AGENT_EVENT:" --all --format="%C(yellow)%h%C(reset) %C(green)%ad%C(reset) %s" --date=relative 2>/dev/null | grep -i "$keyword" >> $temp_file

    # Search in git notes
    git notes --ref=agent-coordination show HEAD 2>/dev/null | grep -i "$keyword" | sed "s/^/📝 /" >> $temp_file

    if test -s $temp_file
        head -20 $temp_file
    else
        echo "   No matches found for '$keyword'"
    end

    rm -f $temp_file
end

function claude-agents-stats
    echo "📊 Agent Coordination Statistics:"
    echo "════════════════════════════════"

    echo "📈 Event Types (last 50 events):"
    set temp_file (mktemp)

    git log -50 --grep="AGENT_EVENT:" --all --format="%s" 2>/dev/null | grep -oE "AGENT_EVENT: [a-z_]+" | sed 's/AGENT_EVENT: //' | sort | uniq -c | sort -nr >> $temp_file
    git notes --ref=agent-coordination show HEAD 2>/dev/null | tail -50 | grep -oE ": [A-Z_]+ \|" | sed -E 's/^: //; s/ \|$//' | sort | uniq -c | sort -nr >> $temp_file

    if test -s $temp_file
        head -10 $temp_file
    end

    echo
    echo "🤖 Most Active Agents:"
    set temp_file2 (mktemp)

    git log -50 --grep="AGENT_EVENT:" --all --format="%s" 2>/dev/null | grep -oE "agent: [^|]+" | sed 's/agent: //; s/ *$//' | sort | uniq -c | sort -nr >> $temp_file2
    git notes --ref=agent-coordination show HEAD 2>/dev/null | tail -50 | grep -oE "agent: [^|]+" | sed 's/agent: //; s/ *$//' | sort | uniq -c | sort -nr >> $temp_file2

    if test -s $temp_file2
        head -5 $temp_file2
    end

    rm -f $temp_file $temp_file2
end

function claude-agents-cleanup
    echo "🧹 Cleaning up old agent coordination data..."
    echo "═══════════════════════════════════════════"

    # Remove agent event commits older than 30 days
    echo "Removing agent event commits older than 30 days..."
    git log --since="30 days ago" --until="now" --grep="AGENT_EVENT:" --format="%H" --all | head -100 > /tmp/keep_commits.txt

    # Clean up git notes (keep last 1000 entries)
    echo "Trimming coordination notes to last 1000 entries..."
    set notes_file /tmp/coordination_notes_clean.txt
    git notes --ref=agent-coordination show HEAD 2>/dev/null | tail -1000 > $notes_file

    if test -s $notes_file
        git notes --ref=agent-coordination remove HEAD 2>/dev/null; or true
        cat $notes_file | git notes --ref=agent-coordination add HEAD
    end

    echo "✅ Cleanup complete!"
    rm -f /tmp/keep_commits.txt $notes_file
end

# ---------- internal helpers ----------

function _claude_coord_dir
    set -l root (git rev-parse --show-toplevel 2>/dev/null; or echo $PWD)
    echo "$root/.claude/coordination"
end

function _claude_coord_init
    set -l dir (_claude_coord_dir)
    mkdir -p "$dir/locks" 2>/dev/null
    test -f "$dir/tasks.json"   ; or echo '{"tasks": []}' > "$dir/tasks.json"
    test -f "$dir/messages.log" ; or touch "$dir/messages.log"
end

function _claude_agent_id
    if set -q CLAUDE_AGENT_ID
        echo $CLAUDE_AGENT_ID
    else
        set -l base "claude-"(whoami)"-"(hostname -s 2>/dev/null; or echo local)
        if set -q CLAUDE_SESSION_ID
            echo "$base-$CLAUDE_SESSION_ID"
        else
            echo $base
        end
    end
end

function _claude_log_event
    set -l kind $argv[1]
    set -e argv[1]
    set -l ts (date -Iseconds)
    set -l agent (_claude_agent_id)
    git notes --ref=agent-coordination append -m "$ts: $kind | agent: $agent | $argv" HEAD 2>/dev/null; or true
end

# ---------- new: status / monitor / analyze ----------

function claude-agents-status
    _claude_coord_init
    echo "🤖 Agent Coordination Status"
    echo "════════════════════════════"
    echo "You: "(_claude_agent_id)
    echo
    echo "Active agents (last 24h):"
    claude-agents-active
    echo
    echo "Open tasks:"
    claude-agents-tasks open
end

function claude-agents-monitor
    echo "🔭 Live coordination monitor (Ctrl-C to exit)"
    set -l last ""
    while true
        set -l now (git notes --ref=agent-coordination show HEAD 2>/dev/null | tail -5)
        if test "$now" != "$last"
            echo "── "(date '+%H:%M:%S')" ──"
            echo $now
            set last $now
        end
        sleep 5
    end
end

function claude-agents-analyze
    set -l root (git rev-parse --show-toplevel 2>/dev/null; or echo $PWD)
    echo "🔍 Repository: "(basename $root)
    echo "Path: $root"
    if test -f "$root/package.json"
        echo "Type: node"
    else if test -f "$root/pyproject.toml" -o -f "$root/requirements.txt"
        echo "Type: python"
    else if test -f "$root/go.mod"
        echo "Type: go"
    else if test -f "$root/Cargo.toml"
        echo "Type: rust"
    else
        echo "Type: generic"
    end
    echo "Branch: "(git -C $root branch --show-current 2>/dev/null)
    echo "Tracked files: "(git -C $root ls-files 2>/dev/null | wc -l)
end

# ---------- new: tasks ----------

function claude-agents-tasks
    _claude_coord_init
    set -l filter all
    if test (count $argv) -gt 0
        set filter $argv[1]
    end
    set -l file (_claude_coord_dir)/tasks.json
    if not type -q jq
        echo "jq required for task commands" >&2; return 1
    end
    set -l me (_claude_agent_id)
    echo "📝 Tasks ($filter):"
    switch $filter
        case open
            jq -r '.tasks[] | select(.status=="open") | "  [\(.id)] \(.description) — assignee: \(.assignee // "—") deadline: \(.deadline // "—")"' $file
        case done
            jq -r '.tasks[] | select(.status=="done") | "  [\(.id)] \(.description) — done by \(.assignee // "—")"' $file
        case mine
            jq -r --arg me "$me" '.tasks[] | select(.assignee==$me) | "  [\(.id)] \(.description) — \(.status)"' $file
        case '*'
            jq -r '.tasks[] | "  [\(.id)] (\(.status)) \(.description) — assignee: \(.assignee // "—")"' $file
    end
end

function claude-agents-task-add
    if test (count $argv) -lt 2
        echo "Usage: claude-agents-task-add <id> <description> [deadline-iso]"; return 1
    end
    _claude_coord_init
    set -l id $argv[1]; set -l desc $argv[2]
    set -l deadline ""
    if test (count $argv) -ge 3
        set deadline $argv[3]
    end
    set -l file (_claude_coord_dir)/tasks.json
    if jq -e --arg id "$id" '.tasks[] | select(.id==$id)' $file >/dev/null
        echo "❌ Task $id already exists"; return 1
    end
    set -l tmp (mktemp)
    jq --arg id "$id" --arg desc "$desc" --arg deadline "$deadline" --arg ts (date -Iseconds) \
        '.tasks += [{id:$id, description:$desc, deadline:$deadline, status:"open", assignee:null, created_at:$ts}]' \
        $file > $tmp; and mv $tmp $file
    _claude_log_event TASK_ADDED "task: $id | description: $desc"
    echo "✅ Task $id added"
end

function claude-agents-assign
    if test (count $argv) -eq 0
        echo "Usage: claude-agents-assign <task-id> [deadline-iso]"; return 1
    end
    _claude_coord_init
    set -l id $argv[1]
    set -l deadline ""
    if test (count $argv) -ge 2
        set deadline $argv[2]
    end
    set -l file (_claude_coord_dir)/tasks.json
    set -l me (_claude_agent_id)
    if not jq -e --arg id "$id" '.tasks[] | select(.id==$id)' $file >/dev/null
        echo "❌ Task $id not found"; return 1
    end
    set -l tmp (mktemp)
    jq --arg id "$id" --arg me "$me" --arg deadline "$deadline" \
        '(.tasks[] | select(.id==$id)) |= (.assignee=$me | (if $deadline=="" then . else .deadline=$deadline end))' \
        $file > $tmp; and mv $tmp $file
    _claude_log_event TASK_ASSIGNED "task: $id | deadline: $deadline"
    echo "✅ $me assigned to $id"
end

function claude-agents-task-done
    if test (count $argv) -eq 0
        echo "Usage: claude-agents-task-done <task-id>"; return 1
    end
    _claude_coord_init
    set -l id $argv[1]
    set -l file (_claude_coord_dir)/tasks.json
    set -l tmp (mktemp)
    jq --arg id "$id" --arg ts (date -Iseconds) \
        '(.tasks[] | select(.id==$id)) |= (.status="done" | .completed_at=$ts)' \
        $file > $tmp; and mv $tmp $file
    _claude_log_event TASK_DONE "task: $id"
    echo "✅ Task $id marked done"
end

# ---------- new: messaging ----------

function claude-agents-broadcast
    if test (count $argv) -eq 0
        echo "Usage: claude-agents-broadcast <message>"; return 1
    end
    _claude_coord_init
    set -l me (_claude_agent_id)
    set -l ts (date -Iseconds)
    set -l msg "$argv"
    echo "$ts | BROADCAST | from: $me | $msg" >> (_claude_coord_dir)/messages.log
    _claude_log_event BROADCAST "$msg"
    echo "📣 broadcast sent"
end

function claude-agents-dm
    if test (count $argv) -lt 2
        echo "Usage: claude-agents-dm <agent-id> <message>"; return 1
    end
    _claude_coord_init
    set -l to $argv[1]
    set -e argv[1]
    set -l msg "$argv"
    set -l me (_claude_agent_id)
    set -l ts (date -Iseconds)
    echo "$ts | DM | from: $me | to: $to | $msg" >> (_claude_coord_dir)/messages.log
    _claude_log_event DM "to: $to | $msg"
    echo "✉️  dm to $to sent"
end

function claude-agents-inbox
    _claude_coord_init
    set -l me (_claude_agent_id)
    set -l file (_claude_coord_dir)/messages.log
    echo "📥 Messages for $me (broadcasts + DMs to you):"
    grep -E "BROADCAST|to: $me " $file 2>/dev/null | tail -50; or echo "   (empty)"
end

# ---------- locks ----------

function claude-agents-locks
    _claude_coord_init
    set -l dir (_claude_coord_dir)/locks
    mkdir -p $dir 2>/dev/null
    set -l now (date +%s)
    set -l ttl 600
    if set -q CLAUDE_LOCK_TTL; set ttl $CLAUDE_LOCK_TTL; end
    echo "🔒 Active advisory locks (TTL "$ttl"s):"
    set -l found 0
    for lock in $dir/*.lock
        test -f $lock; or continue
        set -l agent (grep '^agent=' $lock | cut -d= -f2-)
        set -l path (grep '^path=' $lock | cut -d= -f2-)
        set -l ts_epoch (grep '^ts_epoch=' $lock | cut -d= -f2-)
        set -l age 0
        test -n "$ts_epoch"; and set age (math $now - $ts_epoch)
        set -l stale ""
        test $age -ge $ttl; and set stale " (stale)"
        printf "  %s\n    agent=%s  age=%ss%s\n" $path $agent $age $stale
        set found 1
    end
    test $found -eq 0; and echo "  (none)"
end

function claude-agents-locks-clear
    _claude_coord_init
    set -l dir (_claude_coord_dir)/locks
    mkdir -p $dir 2>/dev/null
    set -l mode stale
    if test (count $argv) -gt 0
        set mode $argv[1]
    end
    set -l now (date +%s)
    set -l ttl 600
    if set -q CLAUDE_LOCK_TTL; set ttl $CLAUDE_LOCK_TTL; end
    set -l me (_claude_agent_id)
    set -l removed 0
    for lock in $dir/*.lock
        test -f $lock; or continue
        set -l agent (grep '^agent=' $lock | cut -d= -f2-)
        set -l ts_epoch (grep '^ts_epoch=' $lock | cut -d= -f2-)
        set -l age 0
        test -n "$ts_epoch"; and set age (math $now - $ts_epoch)
        switch $mode
            case all
                rm -f $lock; set removed (math $removed + 1)
            case mine
                if test "$agent" = "$me"
                    rm -f $lock; set removed (math $removed + 1)
                end
            case '*'
                if test $age -ge $ttl
                    rm -f $lock; set removed (math $removed + 1)
                end
        end
    end
    echo "🧹 cleared $removed lock(s) (mode=$mode)"
end