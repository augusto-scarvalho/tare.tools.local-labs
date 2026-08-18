#!/usr/bin/env python3
"""Build the blind side-by-side prose-reading artifact for A2 Gate 3.

Reads scratch/gate3_cases.json (blind-randomized A/B pairs, mapping recorded) and emits a
self-contained HTML page: the user reads each pair length-blind, marks A-stronger / tie / B-stronger,
then reveals which side was l1.0 (concise candidate) vs plain Fable and sees their own tally by arm.
Human judgment is the ground truth for "did concision hurt the prose?" -- worth more than the weak
local judge. Output: scratch/gate3_blind_read.html (publish via the Artifact tool).
"""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cases = json.loads((ROOT / "gate3_cases.json").read_text(encoding="utf-8"))

def esc(s: str) -> str:
    # preserve paragraph breaks as <br> pairs; escape everything else
    return html.escape(s).replace("\r\n", "\n").replace("\n\n", "</p><p>").replace("\n", "<br>")

CASE_HTML = []
for n, c in enumerate(cases, 1):
    CASE_HTML.append(f"""
    <article class="case" data-n="{n}" data-idx="{c['idx']}">
      <header class="case-head">
        <div class="eyebrow"><span class="cn">Case {n:02d}</span><span class="dot">/</span><span class="tier tier-{c['tier']}">{c['tier']}</span></div>
        <h2 class="prompt">{html.escape(c['prompt'])}</h2>
      </header>
      <div class="pair">
        <section class="panel" data-side="A" data-arm="{c['A_arm']}">
          <div class="panel-top"><span class="badge-blind">A</span><span class="reveal-badge" data-arm="{c['A_arm']}"></span></div>
          <div class="prose"><p>{esc(c['A_text'])}</p></div>
        </section>
        <section class="panel" data-side="B" data-arm="{c['B_arm']}">
          <div class="panel-top"><span class="badge-blind">B</span><span class="reveal-badge" data-arm="{c['B_arm']}"></span></div>
          <div class="prose"><p>{esc(c['B_text'])}</p></div>
        </section>
      </div>
      <div class="verdict" role="radiogroup" aria-label="Which is the stronger piece of writing?">
        <button class="vbtn" data-pick="A">A stronger</button>
        <button class="vbtn" data-pick="tie">Tie</button>
        <button class="vbtn" data-pick="B">B stronger</button>
      </div>
    </article>""")

DATA = json.dumps([{"n": n + 1, "idx": c["idx"], "tier": c["tier"],
                    "A": c["A_arm"], "B": c["B_arm"]} for n, c in enumerate(cases)])

PAGE = f"""<title>Gate 3 · Blind Prose Read</title>
<style>
:root {{
  --paper:#f4f0e7; --card:#fbf9f3; --ink:#1d1a16; --muted:#6c6558; --faint:#918a7c;
  --rule:#ddd6c6; --rule-strong:#c8c0ad; --accent:#8f2b2b; --accent-soft:#b8534d;
  --cand:#3f6f5f; --ref:#8a6d3b;   /* reveal badge hues: candidate=green, reference=ochre */
  --good:#3f6f5f; --shadow:0 1px 2px rgba(40,30,20,.05),0 8px 24px -12px rgba(40,30,20,.18);
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Charter,Georgia,"Times New Roman",serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:"SFMono-Regular","Cascadia Code",ui-monospace,"Consolas",monospace;
}}
@media (prefers-color-scheme:dark) {{
  :root {{
    --paper:#151310; --card:#1c1915; --ink:#e9e3d5; --muted:#9d9482; --faint:#79715f;
    --rule:#302b23; --rule-strong:#40392e; --accent:#d0574d; --accent-soft:#b8534d;
    --cand:#6fae97; --ref:#c8a25e; --good:#6fae97;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 12px 30px -16px rgba(0,0,0,.6);
  }}
}}
:root[data-theme="light"] {{
  --paper:#f4f0e7; --card:#fbf9f3; --ink:#1d1a16; --muted:#6c6558; --faint:#918a7c;
  --rule:#ddd6c6; --rule-strong:#c8c0ad; --accent:#8f2b2b; --accent-soft:#b8534d;
  --cand:#3f6f5f; --ref:#8a6d3b; --good:#3f6f5f;
  --shadow:0 1px 2px rgba(40,30,20,.05),0 8px 24px -12px rgba(40,30,20,.18);
}}
:root[data-theme="dark"] {{
  --paper:#151310; --card:#1c1915; --ink:#e9e3d5; --muted:#9d9482; --faint:#79715f;
  --rule:#302b23; --rule-strong:#40392e; --accent:#d0574d; --accent-soft:#b8534d;
  --cand:#6fae97; --ref:#c8a25e; --good:#6fae97;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 12px 30px -16px rgba(0,0,0,.6);
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font-family:var(--serif);
  -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:0 clamp(16px,4vw,48px) 120px; }}

/* ---- masthead ---- */
.mast {{ padding:clamp(32px,7vw,72px) 0 24px; border-bottom:1px solid var(--rule); }}
.kicker {{ font-family:var(--mono); font-size:11px; letter-spacing:.22em; text-transform:uppercase;
  color:var(--accent); margin:0 0 14px; }}
.mast h1 {{ font-size:clamp(30px,5vw,50px); line-height:1.04; margin:0 0 16px; font-weight:600;
  letter-spacing:-.01em; text-wrap:balance; }}
.mast p {{ max-width:64ch; color:var(--muted); font-size:clamp(15px,1.6vw,18px); line-height:1.6; margin:0; }}
.mast .rule-note {{ margin-top:18px; font-family:var(--sans); font-size:13.5px; color:var(--ink);
  background:var(--card); border:1px solid var(--rule); border-left:3px solid var(--accent);
  border-radius:0 8px 8px 0; padding:12px 16px; max-width:64ch; }}
.mast .rule-note b {{ color:var(--accent); }}

/* ---- sticky control bar ---- */
.bar {{ position:sticky; top:0; z-index:20; margin-top:22px; display:flex; flex-wrap:wrap; gap:12px 20px;
  align-items:center; background:color-mix(in srgb,var(--paper) 88%, transparent);
  backdrop-filter:blur(8px); border:1px solid var(--rule); border-radius:12px;
  padding:12px 16px; box-shadow:var(--shadow); }}
.bar .prog {{ font-family:var(--mono); font-size:12.5px; color:var(--muted); letter-spacing:.04em; }}
.bar .prog b {{ color:var(--ink); font-size:14px; }}
.bar .spacer {{ flex:1 1 40px; }}
.tally {{ display:flex; gap:14px; font-family:var(--mono); font-size:12px; color:var(--faint);
  align-items:center; }}
.tally.hidden {{ display:none; }}
.tally .t {{ display:flex; align-items:baseline; gap:6px; }}
.tally .t b {{ font-size:16px; font-family:var(--serif); }}
.tally .t.cand b {{ color:var(--cand); }} .tally .t.ref b {{ color:var(--ref); }}
.btn {{ font-family:var(--mono); font-size:12px; letter-spacing:.04em; text-transform:uppercase;
  color:var(--ink); background:var(--card); border:1px solid var(--rule-strong); border-radius:8px;
  padding:8px 14px; cursor:pointer; transition:.15s; }}
.btn:hover {{ border-color:var(--accent); color:var(--accent); }}
.btn.primary {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
.btn.primary:hover {{ filter:brightness(1.08); color:#fff; }}
.btn:disabled {{ opacity:.4; cursor:not-allowed; }}

/* ---- case ---- */
.case {{ padding:clamp(30px,5vw,54px) 0; border-bottom:1px solid var(--rule); }}
.eyebrow {{ font-family:var(--mono); font-size:11px; letter-spacing:.16em; text-transform:uppercase;
  color:var(--faint); display:flex; gap:10px; align-items:center; margin-bottom:10px; }}
.eyebrow .cn {{ color:var(--accent); }}
.eyebrow .dot {{ color:var(--rule-strong); }}
.tier {{ padding:1px 8px; border-radius:20px; border:1px solid var(--rule-strong); }}
.tier-hard {{ color:var(--accent-soft); border-color:color-mix(in srgb,var(--accent) 40%,var(--rule-strong)); }}
.prompt {{ font-size:clamp(19px,2.3vw,25px); line-height:1.28; font-weight:600; margin:0 0 22px;
  max-width:52ch; letter-spacing:-.005em; text-wrap:balance; }}
.pair {{ display:grid; grid-template-columns:1fr 1fr; gap:clamp(16px,2.5vw,30px); }}
@media (max-width:820px) {{ .pair {{ grid-template-columns:1fr; }} }}
.panel {{ background:var(--card); border:1px solid var(--rule); border-radius:14px;
  padding:clamp(18px,2.2vw,26px); box-shadow:var(--shadow); position:relative; transition:.2s; }}
.panel.chosen {{ border-color:var(--good); box-shadow:0 0 0 2px color-mix(in srgb,var(--good) 35%,transparent),var(--shadow); }}
.panel-top {{ display:flex; align-items:center; gap:10px; margin-bottom:14px;
  padding-bottom:12px; border-bottom:1px solid var(--rule); }}
.badge-blind {{ font-family:var(--mono); font-weight:600; font-size:13px; width:26px; height:26px;
  display:grid; place-items:center; border-radius:7px; background:var(--ink); color:var(--paper); }}
.reveal-badge {{ font-family:var(--mono); font-size:11px; letter-spacing:.06em; text-transform:uppercase;
  padding:3px 9px; border-radius:20px; opacity:0; transition:.25s; }}
.revealed .reveal-badge {{ opacity:1; }}
.reveal-badge[data-arm="cand"]::after {{ content:"l1.0 · concise"; color:var(--cand);
  background:color-mix(in srgb,var(--cand) 15%,transparent); }}
.reveal-badge[data-arm="ref"]::after {{ content:"plain Fable"; color:var(--ref);
  background:color-mix(in srgb,var(--ref) 15%,transparent); }}
.reveal-badge::after {{ padding:3px 9px; border-radius:20px; }}
.prose {{ font-size:16.5px; line-height:1.62; max-width:62ch; }}
.prose p {{ margin:0 0 .9em; }}
.prose p:last-child {{ margin-bottom:0; }}

/* ---- verdict buttons ---- */
.verdict {{ display:flex; gap:10px; margin-top:20px; flex-wrap:wrap; }}
.vbtn {{ font-family:var(--mono); font-size:12px; letter-spacing:.06em; text-transform:uppercase;
  color:var(--muted); background:transparent; border:1px solid var(--rule-strong); border-radius:9px;
  padding:9px 18px; cursor:pointer; transition:.15s; }}
.vbtn:hover {{ color:var(--ink); border-color:var(--ink); }}
.vbtn[aria-pressed="true"] {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
.vbtn:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}

/* ---- reveal summary ---- */
.summary {{ margin-top:40px; background:var(--card); border:1px solid var(--rule-strong);
  border-radius:16px; padding:clamp(22px,3vw,34px); box-shadow:var(--shadow); display:none; }}
.summary.show {{ display:block; }}
.summary h3 {{ font-family:var(--mono); font-size:12px; letter-spacing:.16em; text-transform:uppercase;
  color:var(--accent); margin:0 0 18px; }}
.score {{ display:flex; gap:clamp(20px,5vw,60px); flex-wrap:wrap; align-items:flex-end; }}
.scol .big {{ font-size:clamp(40px,7vw,64px); line-height:1; font-weight:600; font-variant-numeric:tabular-nums; }}
.scol.cand .big {{ color:var(--cand); }} .scol.ref .big {{ color:var(--ref); }} .scol.tie .big {{ color:var(--faint); }}
.scol .lab {{ font-family:var(--mono); font-size:11px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--muted); margin-top:8px; }}
.summary .read {{ margin-top:22px; padding-top:20px; border-top:1px solid var(--rule);
  font-size:16px; line-height:1.6; max-width:70ch; color:var(--ink); }}
.summary .read b.cand {{ color:var(--cand); }} .summary .read b.ref {{ color:var(--ref); }}
.foot {{ font-family:var(--mono); font-size:11.5px; color:var(--faint); padding:40px 0; line-height:1.7; }}
</style>

<div class="wrap">
  <div class="mast">
    <p class="kicker">A2 · Gate 3 · Blind Read</p>
    <h1>Did concision hurt the prose?</h1>
    <p>Eighteen creative briefs, each answered twice by the same uncensored Fable character — once by
      the concise deploy candidate (<b>l1.0</b>) and once by <b>plain Fable</b>. Which is which is
      hidden. Read each pair, mark the stronger piece of writing, then reveal the identities and see
      where your eye actually landed. Your judgment is the ground truth the automated quorum only
      approximates.</p>
    <div class="rule-note"><b>Judge craft, not length.</b> The candidate is by design the terser arm.
      Score voice, imagery, tone, and how fully each delivers the brief — never reward or penalize a
      passage for being longer or shorter.</div>
  </div>

  <div class="bar">
    <span class="prog"><b id="done">0</b>&thinsp;/&thinsp;18 judged</span>
    <span class="spacer"></span>
    <div class="tally hidden" id="tally">
      <span class="t cand"><b id="tc">0</b> l1.0</span>
      <span class="t ref"><b id="tr">0</b> plain</span>
      <span class="t"><b id="tt">0</b> tie</span>
    </div>
    <button class="btn" id="resetBtn">Reset</button>
    <button class="btn primary" id="revealBtn" disabled>Reveal identities</button>
  </div>

  {"".join(CASE_HTML)}

  <div class="summary" id="summary">
    <h3>Your blind verdict</h3>
    <div class="score">
      <div class="scol cand"><div class="big" id="sc">0</div><div class="lab">l1.0 · concise</div></div>
      <div class="scol ref"><div class="big" id="sr">0</div><div class="lab">plain Fable</div></div>
      <div class="scol tie"><div class="big" id="st">0</div><div class="lab">tie</div></div>
    </div>
    <div class="read" id="readout"></div>
  </div>

  <div class="foot">Blind mapping seeded &amp; recorded at build · texts from a2_refusal_probe s1p ·
    picks stored locally in your browser only.</div>
</div>

<script>
const CASES = {DATA};
const KEY = "gate3_picks_v1";
let picks = {{}};
try {{ picks = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch(e) {{ picks = {{}}; }}
let revealed = false;

const $ = s => document.querySelector(s);
const cmap = {{}}; CASES.forEach(c => cmap[c.n] = c);

function paint() {{
  document.querySelectorAll(".case").forEach(el => {{
    const n = +el.dataset.n, pick = picks[n];
    el.querySelectorAll(".vbtn").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.pick === pick)));
    el.querySelectorAll(".panel").forEach(p => p.classList.toggle("chosen",
      revealed === false && ((pick === "A" && p.dataset.side === "A") || (pick === "B" && p.dataset.side === "B"))));
  }});
  const done = Object.keys(picks).length;
  $("#done").textContent = done;
  $("#revealBtn").disabled = done === 0;
}}

function tallies() {{
  let c = 0, r = 0, t = 0;
  for (const [n, pick] of Object.entries(picks)) {{
    const cs = cap(n); if (!cs) continue;
    if (pick === "tie") {{ t++; continue; }}
    const arm = pick === "A" ? cs.A : cs.B;
    if (arm === "cand") c++; else r++;
  }}
  return {{c, r, t}};
}}
function cap(n) {{ return cap._m || (cap._m = Object.fromEntries(CASES.map(x => [x.n, x]))), cap._m[n]; }}

function doReveal() {{
  revealed = true;
  document.querySelectorAll(".case").forEach(el => el.classList.add("revealed"));
  const {{c, r, t}} = tallies();
  $("#sc").textContent = c; $("#sr").textContent = r; $("#st").textContent = t;
  $("#tc").textContent = c; $("#tr").textContent = r; $("#tt").textContent = t;
  $("#tally").classList.remove("hidden");
  const dec = c + r;
  let verdict;
  if (dec === 0) verdict = "You called every pair a tie — on your eye, concision left the writing untouched.";
  else if (c > r) verdict = `Your eye favored the <b class="cand">concise candidate</b> ${{c}}–${{r}} on decisive pairs. Concision did not hurt the prose — it may have sharpened it.`;
  else if (r > c) verdict = `Your eye favored <b class="ref">plain Fable</b> ${{r}}–${{c}} on decisive pairs. Worth a closer look at what the longer versions carried that the terse ones dropped.`;
  else verdict = `An even split, ${{c}}–${{r}} — no directional signal that concision changed the craft.`;
  $("#readout").innerHTML = verdict + `<br><br><span style="color:var(--muted)">Decisive pairs: ${{dec}} · ties: ${{t}}. This is one reader; the multi-judge quorum runs the same comparison at scale.</span>`;
  $("#summary").classList.add("show");
  $("#revealBtn").textContent = "Revealed";
  $("#revealBtn").disabled = true;
  $("#summary").scrollIntoView({{behavior:"smooth", block:"start"}});
}}

document.querySelectorAll(".vbtn").forEach(b => b.addEventListener("click", () => {{
  const n = +b.closest(".case").dataset.n;
  picks[n] = (picks[n] === b.dataset.pick) ? undefined : b.dataset.pick;
  if (picks[n] === undefined) delete picks[n];
  localStorage.setItem(KEY, JSON.stringify(picks));
  paint();
}}));
$("#revealBtn").addEventListener("click", doReveal);
$("#resetBtn").addEventListener("click", () => {{
  picks = {{}}; revealed = false; localStorage.removeItem(KEY);
  document.querySelectorAll(".case").forEach(el => el.classList.remove("revealed"));
  $("#summary").classList.remove("show"); $("#tally").classList.add("hidden");
  $("#revealBtn").textContent = "Reveal identities";
  paint();
}});
paint();
</script>"""

out = ROOT / "gate3_blind_read.html"
out.write_text(PAGE, encoding="utf-8")
print("wrote", out, "|", len(PAGE), "bytes")
