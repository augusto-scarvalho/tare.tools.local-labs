import os, re, sys

sys.stdout.reconfigure(encoding='utf-8')

search_dirs = [
    r'C:\projects\tare.tools.local-labs',
    r'C:\projects\_lab_handoff_build',
    r'C:\projects\tare-tools-relay'
]

urls = set()
github_refs = set()
docs_with_links = {}
key_tech_mentions = {}

url_pattern = re.compile(r'https?://[^\s\)\]\"\'\`<>]+')
pr_issue_pattern = re.compile(r'(?:PR|pull/|issues/|issue\s*#?|#)\s*(\d+)', re.IGNORECASE)
paper_arxiv_pattern = re.compile(r'(?:arxiv|abs/|pdf/|doi|biorxiv|huggingface\.co/[^\s]+)', re.IGNORECASE)

tech_keywords = [
    'llama.cpp', 'ik_llama.cpp', 'DSpark', 'EAGLE', 'mergekit', 'SGLang', 'vLLM',
    'DeepCache', 'TeaCache', 'KVarN', 'OSCAR', 'NanoQuant', 'Bielik', 'FA3', 'FlashAttention',
    'MTP', 'Gated Delta Net', 'GDN', 'Mamba', 'Granite', 'LongBench', 'NoLiMa', 'MQAR',
    'ThinkingCap', 'Fable', 'Nemotron', 'Qwen', 'Gemma', 'DavidAU', 'TIES', 'unsloth',
    'vllm', 'sglang', 'sgl-kernel', 'triton', 'torch.compile', 'CUDA graph', 'mmap'
]

for base_dir in search_dirs:
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(('.md', '.py', '.txt', '.sh', '.ps1', '.json', '.html')):
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as fp:
                        content = fp.read()
                        
                        # Find URLs
                        found_urls = url_pattern.findall(content)
                        if found_urls:
                            rel_p = os.path.relpath(filepath, r'C:\projects')
                            docs_with_links[rel_p] = len(found_urls)
                            for u in found_urls:
                                urls.add((u, rel_p))
                                
                        # Find PRs/Issues
                        found_prs = pr_issue_pattern.findall(content)
                        for pr in found_prs:
                            github_refs.add((pr, os.path.basename(filepath)))
                            
                        # Keywords
                        for kw in tech_keywords:
                            if kw.lower() in content.lower():
                                key_tech_mentions[kw] = key_tech_mentions.get(kw, 0) + 1
                except Exception as e:
                    pass

print(f"=== Total Unique URLs Found: {len(urls)} ===")
for u, doc in sorted(list(urls)):
    print(f"[{doc}] -> {u}")

print("\n=== Tech Keywords Frequency ===")
for kw, count in sorted(key_tech_mentions.items(), key=lambda x: x[1], reverse=True):
    print(f"{kw}: {count} occurrences")
