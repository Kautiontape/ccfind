# ccfind

Search and read your old Claude Code sessions. It's read-only: it never resumes,
edits or deletes a session. When you find the one you want, copy its resume
command and paste it into a terminal.

```console
$ ccfind steam networking
```

That opens an fzf search over every prompt, reply, thought, tool call and tool
output in every session, including subagents.

## Install

Needs Python 3.11+ and fzf. Copying uses `wl-copy`, `xclip` or `xsel`.

```sh
git clone https://github.com/Kautiontape/ccfind
ln -s "$PWD/ccfind/ccfind.py" ~/.local/bin/ccfind
```

## Usage

```sh
ccfind                   # search in fzf
ccfind web               # search in the browser at http://127.0.0.1:8977
ccfind search <query>    # print results, --json for scripts
ccfind show <id>         # print a transcript
```

In fzf, `enter` opens the transcript in less (`n`/`N` jumps between matches),
`ctrl-y` copies `cd <dir> && claude --resume <id>`, `alt-i` and `alt-p` copy the
session id and file path, and `alt-w` opens the session in the browser.

## Search

| Query | Finds |
|---|---|
| `deploy token` | both words, as prefixes ("deploy" finds "deployment") |
| `"exact phrase"` | the phrase |
| `redis -docker` | redis, but not docker |
| `nginx OR caddy` | either |
| `p:myproject` | sessions whose path contains it |
| `in:you` | only your prompts (also `claude`, `thinking`, `tools`, `output`) |
| `since:2w` `before:2026-05-01` | sessions in that window |
| `prompts:5` `tokens:50k` | at least that big (`prompts:<3` for at most) |
| `sort:new` `sort:big` | newest or largest first |

Every session shows how many prompts you sent and how many tokens went through
it, so the one-liners are easy to skip.

## How it works

The first run indexes `~/.claude/projects` into a SQLite full-text index at
`~/.cache/ccfind/index.db`. For 4 GB of transcripts that took about 30 seconds
and 800 MB. After that, only changed files are re-read. Transcripts are opened
read-only. The web server binds to 127.0.0.1 and answers GET requests only.

Token counts measure how much the context grew between API calls. That leaves
out the fixed system-prompt overhead and isn't inflated by prompt caching.

Claude Code deletes transcripts after `cleanupPeriodDays` (30 by default). Raise
it in `~/.claude/settings.json` to keep old sessions searchable.
