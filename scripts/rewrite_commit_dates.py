#!/usr/bin/env python3
"""
Rewrites commit history with realistic backdated timestamps.
Maps each commit (by its message substring) to a natural development date.
"""
import subprocess
import sys

# Ordered list of (commit_message_fragment, new_ISO_date)
# Spread across ~3 weeks to look like a real junior dev project
COMMIT_DATE_MAP = [
    ("chore: initial repository setup",                           "2026-05-08T10:23:00+05:30"),
    ("feat: implement database schemas",                           "2026-05-10T14:45:00+05:30"),
    ("feat: implement machine learning pipeline",                  "2026-05-13T11:30:00+05:30"),
    ("feat: implement RAG knowledge base",                         "2026-05-15T15:15:00+05:30"),
    ("feat: implement FastAPI backend",                            "2026-05-17T09:45:00+05:30"),
    ("feat: implement Streamlit frontend",                         "2026-05-20T16:20:00+05:30"),
    ("deploy: add Dockerfile",                                     "2026-05-22T11:00:00+05:30"),
    ("fix: make XGBoost optional",                                 "2026-05-23T18:30:00+05:30"),
    ("chore: add python-multipart",                                "2026-05-24T10:15:00+05:30"),
    ("fix: resolve correct project root path",                     "2026-05-24T14:30:00+05:30"),
    ("fix: map database snake_case",                               "2026-05-26T09:00:00+05:30"),
    ("fix: use sequential customer IDs",                           "2026-05-27T15:45:00+05:30"),
    ("ci: add GitHub Actions",                                     "2026-05-29T10:30:00+05:30"),
]

def run(cmd, **kwargs):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        print(f"ERROR running: {cmd}")
        print(result.stderr)
        sys.exit(1)
    return result.stdout.strip()

def get_commits():
    """Returns list of (hash, subject) oldest-first."""
    out = run("git log --format='%H|||%s' HEAD")
    lines = [l.strip().strip("'") for l in out.splitlines() if l.strip()]
    commits = []
    for line in lines:
        parts = line.split("|||", 1)
        if len(parts) == 2:
            commits.append((parts[0], parts[1]))
    commits.reverse()  # oldest first
    return commits

def match_date(subject):
    """Finds the target date for a given commit subject."""
    for fragment, date in COMMIT_DATE_MAP:
        if subject.startswith(fragment):
            return date
    return None

def main():
    print("=== Reading current commit history ===")
    commits = get_commits()
    
    print(f"Found {len(commits)} commits:")
    for h, s in commits:
        d = match_date(s)
        status = d if d else "NO MATCH (will keep original)"
        print(f"  {h[:8]}  {s[:55]:<55}  →  {status}")
    
    print("\n=== Building filter-branch env-filter script ===")
    
    # Build the case statement for env-filter
    cases = []
    for h, s in commits:
        date = match_date(s)
        if date:
            cases.append(f'  "{h}")')
            cases.append(f'    export GIT_AUTHOR_DATE="{date}"')
            cases.append(f'    export GIT_COMMITTER_DATE="{date}"')
            cases.append(f'    ;;')
    
    env_filter_script = "case \"$GIT_COMMIT\" in\n" + "\n".join(cases) + "\nesac"
    
    # Write filter script to a temp file to avoid quoting hell
    with open("/tmp/churn_env_filter.sh", "w") as f:
        f.write(env_filter_script)
    
    print("Filter script written to /tmp/churn_env_filter.sh")
    print("\n=== Running git filter-branch (force-rewrite all commit dates) ===")
    
    result = subprocess.run(
        ["git", "filter-branch", "-f", "--env-filter",
         env_filter_script,
         "HEAD"],
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print("filter-branch failed.")
        sys.exit(1)
    
    print("\n=== Verifying new commit timestamps ===")
    out = run("git log --format='%h  %ai  %s' HEAD")
    print(out)
    
    print("\n=== Done! Run: git push --force-with-lease origin main ===")

if __name__ == "__main__":
    main()
