# Deployment & Operations — Trading Bot

This document collects every command and process to deploy, update, troubleshoot, and operate the trading bot on your server and locally. Follow steps carefully; treat `.env` as secret and never push it to GitHub.

**Quick Start (fresh server deploy)**
- **Stop service:**
```bash
systemctl stop trading-bot
```
- **Switch to root and remove old install (optional fresh clone):**
```bash
cd /root
rm -rf hyperliquid-trading-agent
```
- **Clone repo:**
```bash
git clone https://github.com/sharath2525/Trading-bot.git hyperliquid-trading-agent
cd hyperliquid-trading-agent
```
- **Create / edit `.env`:**
```bash
nano .env   # paste API keys and settings, save
```
- **Start service:**
```bash
systemctl start trading-bot
```
- **Verify:**
```bash
systemctl status trading-bot
journalctl -u trading-bot -f
```

**Update existing server (recommended day-to-day)**
- On your PC: push your changes to `main`:
```bash
# on Windows dev machine
cd "D:\insta ai bot test\hyperliquid-trading-agent-master"
git add .
git commit -m "describe changes"
git push origin main
```
- On server: pull and restart:
```bash
ssh root@209.38.120.100
cd /root/hyperliquid-trading-agent
git pull origin main
systemctl restart trading-bot
journalctl -u trading-bot -f
```

**If `git pull` fails due to local runtime files (conflict)**
```bash
# keep uncommitted runtime files out of way
git stash
git pull origin main
git stash drop
systemctl restart trading-bot
```

**If you need an exact fresh clone of a different repo path**
```bash
cd /root
rm -rf hyperliquid-trading-agent
git clone https://github.com/sharath2525/Trading-bot.git hyperliquid-trading-agent
cd hyperliquid-trading-agent
nano .env
systemctl start trading-bot
```

**Copy single files from Windows to server (when merging 3rd-party + local edits)**
- Use `scp` to overwrite individual files on the server:
```bash
# from Windows PowerShell (adjust paths)
scp "D:\insta ai bot test\hyperliquid-trading-agent-master\src\main.py" root@209.38.120.100:/root/hyperliquid-trading-agent/src/main.py
scp "D:\insta ai bot test\hyperliquid-trading-agent-master\src\agent\decision_maker.py" root@209.38.120.100:/root/hyperliquid-trading-agent/src/agent/decision_maker.py
```
- After `scp`: ssh in and restart the service:
```bash
ssh root@209.38.120.100
systemctl restart trading-bot
journalctl -u trading-bot -f
```

**Local run for testing**
```bash
# run bot locally (use python3 if default maps differently)
cd "D:\insta ai bot test\hyperliquid-trading-agent-master"
python src/main.py
```

**Service & logging commands (every operator must know)**
- Start/stop/restart:
```bash
systemctl start trading-bot
systemctl stop trading-bot
systemctl restart trading-bot
```
- Status:
```bash
systemctl status trading-bot
ps aux | grep main.py | grep -v grep
```
- Live logs (watch):
```bash
journalctl -u trading-bot -f   # follow live
journalctl -u trading-bot -n 50  # last 50 lines
```
- Filter logs for keywords:
```bash
journalctl -u trading-bot | grep -i "error\|exception\|failed\|critical"
journalctl -u trading-bot | grep "TRADE"
```

**Files to watch (in repo root or working dir)**
- `diary.jsonl` — trade open/close events
- `decisions.jsonl` — per-cycle summaries
- `llm_requests.log` — LLM/Claude calls and costs
- `prompts.log` — prompts sent to Claude

Tail these in real time:
```bash
tail -f diary.jsonl
tail -f decisions.jsonl
tail -f llm_requests.log
```

**Clear large logs / truncate JSON files**
- When disk fills, truncate safely (this empties contents but keeps files):
```bash
> diary.jsonl
> decisions.jsonl
> llm_requests.log
> prompts.log
```

**Dashboard & API endpoints**
- Live JSON endpoints (web):
  - `http://209.38.120.100:3000/live`
  - `http://209.38.120.100:3000/diary` (use `?limit=` query)
  - `http://209.38.120.100:3000/logs`

**Daily maintenance checklist**
- Confirm service running:
```bash
systemctl status trading-bot
```
- Check latest cycles in `decisions.jsonl`:
```bash
tail -20 decisions.jsonl
```
- Verify LLM calls and costs:
```bash
tail -50 llm_requests.log
```
- Confirm dashboard reachable in browser: `http://209.38.120.100:3000`

**Merging branches & keeping master/main tidy**
- Typical safe flow:
```bash
# ensure main is tested locally and pushed
git checkout main
git pull origin main
# create or update master from main
git switch -c master main
# or merge main into existing master
git switch master
git pull origin master
git merge main
git push origin master
```
- Avoid committing runtime logs and `.env` — ensure `.gitignore` contains:
```
decisions.jsonl
prompts.log
llm_requests.log
.env
node_modules/
```

**Rollback strategy**
- If a new push breaks things, revert via git:
```bash
# quick rollback to previous commit on server
cd /root/hyperliquid-trading-agent
git log --oneline   # find last good commit hash
git checkout <good-hash>
systemctl restart trading-bot
# to go back to main branch later
git checkout main
```
- Or use `git revert <commit>` to create a reversing commit and push from your PC.

**Backing up server .env before pulls**
```bash
cp /root/hyperliquid-trading-agent/.env /root/.env.backup.$(date +%F_%T)
```

**If the server has files created at runtime that you do not want tracked**
- Use `.gitignore` to prevent accidental tracking. If previously tracked, remove from git tracking:
```bash
git rm --cached decisions.jsonl
git commit -m "stop tracking runtime logs"
git push origin main
```

**Common problems & fixes**
- Git pull conflict: use `git stash` then `git pull` then `git stash drop`.
- Service fails to start: check `journalctl -u trading-bot -n 200` for stack traces.
- Missing Python dependencies: ensure virtualenv or system Python has requirements from `requirements.txt`.
- Port 3000 dashboard not reachable: ensure firewall allows traffic and the bot binds to 0.0.0.0.

**Commands summary — copyable**
```bash
# Deploy update (fast)
ssh root@209.38.120.100
cd /root/hyperliquid-trading-agent
git pull origin main
systemctl restart trading-bot
journalctl -u trading-bot -f

# Fresh reinstall
ssh root@209.38.120.100
systemctl stop trading-bot
cd /root
rm -rf hyperliquid-trading-agent
git clone https://github.com/sharath2525/Trading-bot.git hyperliquid-trading-agent
cd hyperliquid-trading-agent
nano .env
systemctl start trading-bot

# Copy single file from local to server
scp "D:\insta ai bot test\hyperliquid-trading-agent-master\src\main.py" root@209.38.120.100:/root/hyperliquid-trading-agent/src/main.py
scp "D:\insta ai bot test\hyperliquid-trading-agent-master\src\agent\decision_maker.py" root@209.38.120.100:/root/hyperliquid-trading-agent/src/agent/decision_maker.py
ssh root@209.38.120.100
systemctl restart trading-bot
journalctl -u trading-bot -f
```

**Security & best practices**
- Never commit `.env` to Git. Keep it only on the server and your local dev machine.
- Use `scp` or secure editors for secrets.
- Use `git stash` instead of committing runtime files.
- Keep service unit with `Restart=on-failure` so it auto recovers.
- Regularly check `df -h` to ensure logs don't fill disk.

**Where to find files in this workspace**
- Code root: `/root/hyperliquid-trading-agent` on server
- Local workspace: `D:\insta ai bot test\hyperliquid-trading-agent-master`
- Service logs: `journalctl -u trading-bot`
- JSON logs: `diary.jsonl`, `decisions.jsonl`

---
If you want, I can also:
- Add this file to the repo now and commit it for you, or
- Create a shorter quick-reference cheat sheet for a laptop-sized printout.
