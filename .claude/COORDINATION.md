# Claude Multi-Agent Coordination System 🤖

This project is configured for **multi-agent collaboration** where multiple Claude Code agents can coordinate work on the same codebase through git-based communication.

## 🎯 How It Works

The system uses **git commits and notes** as a shared communication channel:
- **Git Notes**: Real-time event logging using `git notes --ref=agent-coordination`
- **Git Commits**: Persistent event markers with structured commit messages
- **Branch Awareness**: Agents can see what branches others are working on
- **File Coordination**: Track which files are being modified by which agents

## 🚀 Quick Start

### 1. Source the helper functions
```bash
source .claude/agent-coordination-helpers.sh
```

### 2. View recent agent activity
```bash
claude-agents-log
```

### 3. See who's currently active
```bash
claude-agents-active
```

## 📋 Available Commands

### Activity & status
| Command | Description |
|---------|-------------|
| `claude-agents-log` | Recent agent activity (notes + commits) |
| `claude-agents-active` | Agents active in last 24h |
| `claude-agents-today` / `-yesterday` | Day-scoped activity |
| `claude-agents-search <kw>` | Search events |
| `claude-agents-stats` | Coordination statistics |
| `claude-agents-status` | Combined overview: agents + open tasks |
| `claude-agents-monitor` | Live tail of coordination notes |
| `claude-agents-analyze` | Quick repo summary |

### Tasks
| Command | Description |
|---------|-------------|
| `claude-agents-tasks [open\|done\|mine\|all]` | List tasks |
| `claude-agents-task-add <id> <desc> [deadline]` | Create task |
| `claude-agents-assign <id> [deadline]` | Take ownership |
| `claude-agents-task-done <id>` | Mark complete |

### Messaging
| Command | Description |
|---------|-------------|
| `claude-agents-broadcast <msg>` | All-agent message |
| `claude-agents-dm <agent-id> <msg>` | Direct message |
| `claude-agents-inbox` | Broadcasts + DMs to you |

### Locks
| Command | Description |
|---------|-------------|
| `claude-agents-locks` | List active advisory locks |
| `claude-agents-locks-clear [stale\|mine\|all]` | Drop locks (default `stale`) |

### Maintenance
| Command | Description |
|---------|-------------|
| `claude-agents-cleanup` | Trim old coordination data |

## 🔍 Event Types Tracked

The system automatically logs these events:

### Session Events
- **SESSION_START**: Agent begins working
- **SESSION_STOP**: Agent finishes working

### Git Operations
- **PRE_COMMIT**: Agent preparing to commit
- **COMMIT_COMPLETE**: Agent finished committing (with hash)
- **PUSH_COMPLETE**: Agent pushed changes
- **MERGE_COMPLETE**: Agent merged branches
- **BRANCH_CREATE**: Agent created new branch

### Task Management
- **TASK_CREATED**: Agent created a new task
- **TASK_COMPLETED**: Agent completed a task

### File Operations
- **FILE_MODIFY_START**: Agent began editing a file
- **FILE_MODIFY_COMPLETE**: Agent finished editing a file

## 🏷️ Agent Identification

Each agent is automatically identified by:
```bash
CLAUDE_AGENT_ID="claude-${USER}-${HOSTNAME}${CLAUDE_SESSION_ID:+-$CLAUDE_SESSION_ID}"
```

Examples:
- `claude-alice-laptop`
- `claude-bob-server01`
- `claude-charlie-dev-machine`

### Running multiple agents on the same host

Two Claude sessions on the same machine produce the same default
`CLAUDE_AGENT_ID` and clash in the coordination log. Disambiguate by exporting
`CLAUDE_SESSION_ID` before launching each session:

```bash
# terminal 1
CLAUDE_SESSION_ID=auth claude
# terminal 2
CLAUDE_SESSION_ID=db claude
```

The helpers and hooks both honor `CLAUDE_SESSION_ID`, so the resulting IDs
become `claude-alice-laptop-auth` and `claude-alice-laptop-db`. You can also
override `CLAUDE_AGENT_ID` directly for full control.

## 🔧 Configuration

The coordination system is configured in `.claude/settings.json` with hooks that trigger on various Claude Code events:

- **SessionStart/Stop**: Log when agents begin/end work
- **PreToolUse/PostToolUse**: Track git operations and file edits
- **TaskCreated/TaskCompleted**: Monitor task lifecycle

## 📊 Example Agent Coordination Flow

```bash
# Agent 1 starts working
SESSION_START | agent: claude-alice-laptop | timestamp: 2026-04-16T10:30:00

# Agent 1 creates a task
TASK_CREATED | agent: claude-alice-laptop | task: implement authentication

# Agent 1 starts editing files
FILE_MODIFY_START | agent: claude-alice-laptop | file: src/auth.js

# Agent 2 starts working and sees Agent 1's activity
SESSION_START | agent: claude-bob-server | timestamp: 2026-04-16T10:35:00

# Agent 2 checks what's happening
$ claude-agents-active
👥 Active Agents (last 24 hours):
   agent: claude-alice-laptop
   agent: claude-bob-server

# Agent 1 commits changes
COMMIT_COMPLETE | agent: claude-alice-laptop | hash: abc123 | branch: feature-auth

# Agent 2 can coordinate to avoid conflicts
$ claude-agents-search "auth"
🔍 Searching agent events for: 'auth'
   TASK_CREATED | agent: claude-alice-laptop | task: implement authentication
   FILE_MODIFY_START | agent: claude-alice-laptop | file: src/auth.js
```

## 🔄 Git Integration

The system integrates with your normal git workflow:

```bash
# All coordination data is stored in git
git log --grep="AGENT_EVENT:" --oneline    # View agent commits
git notes --ref=agent-coordination show    # View coordination notes
git push origin refs/notes/agent-coordination  # Share notes with team
```

## 🛠️ Troubleshooting

### No agent activity showing?
1. Check if hooks are working: `cat .claude/settings.json`
2. Verify git is initialized: `git status`
3. Ensure you have permission to create commits

### Agent ID not showing correctly?
The agent ID uses environment variables. You can override it:
```bash
export CLAUDE_AGENT_ID="claude-your-custom-name"
```

### Want to disable coordination temporarily?
```bash
# Disable all hooks temporarily
export CLAUDE_DISABLE_HOOKS=true
```

## 🚀 Advanced Usage

### Custom Branch Naming for Coordination
Use descriptive branch names that include agent info:
```bash
git checkout -b claude-alice/feature-auth
git checkout -b claude-bob/fix-validation
```

### Share Coordination Notes
```bash
# Push coordination notes to remote
git push origin refs/notes/agent-coordination

# Pull coordination notes from remote
git fetch origin refs/notes/agent-coordination:refs/notes/agent-coordination
```

### Integration with CI/CD
The coordination events can trigger automated workflows:
```bash
# Example: Trigger tests when any agent pushes
if claude-agents-search "PUSH_COMPLETE" | grep -q "$(date +%Y-%m-%d)"; then
    echo "Running tests due to recent agent activity..."
    npm test
fi
```

## 📝 Contributing

To add new event types or coordination features:
1. Edit `.claude/settings.json` to add new hooks
2. Update the helper functions in `.claude/agent-coordination-helpers.sh`
3. Test with multiple agents to ensure coordination works
4. Update this README with new features

## 🤝 Team Setup

For your team to use this system:

1. **Commit the settings**: The `.claude/settings.json` is already set up and should be committed
2. **Share the workflow**: Team members should run `source .claude/agent-coordination-helpers.sh`
3. **Sync coordination data**: Use `git push/pull refs/notes/agent-coordination`
4. **Establish conventions**: Agree on branch naming and coordination patterns

---

**Happy coordinated coding! 🎉**

Need help? Run `claude-agents-log` to see recent activity or `claude-agents-stats` for insights.