import os, re, sys

sys.stdout.reconfigure(encoding='utf-8')

search_files = []

for base_dir in [r'C:\projects\tare.tools.local-labs', r'C:\projects\_lab_handoff_build\local-ai-lab-handoff-2026-08-10']:
    for root, dirs, files in os.walk(base_dir):
        if 'node_modules' in root or '.git' in root or '_lab_handoff_build\\local-ai-lab-handoff-2026-08-10\\05_models' in root:
            continue
        for f in files:
            if f.endswith(('.md', '.py', '.txt')):
                search_files.append(os.path.join(root, f))

print(f"Total files to scan: {len(search_files)}")

urls = []
prs_issues = []
papers_terms = set()

url_reg = re.compile(r'https?://[^\s\)\]\"\'\`<>]+')
pr_reg = re.compile(r'(?:PR|pull/|issues/|issue|#)\s*([0-9]+|[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+#(?:PR)?\d+)', re.IGNORECASE)

external_mentions = []

for path in search_files:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as fp:
            content = fp.read()
            rel = os.path.relpath(path, r'C:\projects')
            
            for u in url_reg.findall(content):
                urls.append((u, rel))
                
            for p in pr_reg.findall(content):
                prs_issues.append((p, rel))
                
            # Look for explicit external references (github repos, papers, authors, forks)
            lines = content.split('\n')
            for idx, line in enumerate(lines):
                if any(term in line.lower() for term in ['github.com', 'arxiv', 'huggingface', 'paper', 'pr #', 'issue #', 'fork', 'commit', 'ik_llama', 'dspark', 'eagle', 'ties', 'mergekit', 'sglang', 'vllm', 'deepcache', 'teacache', 'kvarn', 'oscar', 'nanoquant', 'bielik', 'fa3', 'flashattention', 'mtp', 'gated delta net', 'mamba', 'granite', 'longbench', 'nolima', 'mqar', 'davidau']):
                    external_mentions.append((rel, idx+1, line.strip()))
    except Exception as e:
        pass

print(f"\n=== Found {len(urls)} URLs ===")
for u, rel in urls[:40]:
    print(f"[{rel}] -> {u}")

print(f"\n=== Found {len(prs_issues)} PR/Issue references ===")
for p, rel in prs_issues[:30]:
    print(f"[{rel}] -> #{p}")

print(f"\n=== External Mentions Sample ({len(external_mentions)} lines found) ===")
for rel, line_num, line in external_mentions[:50]:
    print(f"[{rel}:L{line_num}] {line[:120]}")
