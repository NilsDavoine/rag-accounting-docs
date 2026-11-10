"""Small local web page.  uvicorn app.web:app --port 8000

The company selector stands in for authentication: in a real deployment the tenant comes
from the signed-in user's session, never from a field the user can edit."""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag import store
from rag.pipeline import ask

app = FastAPI(title="RAG over accounting documents")
COMPANIES = {"ombrelune": "Ombrelune Logistique (FR)", "quillfern": "Quillfern Software (EN)",
             "halvardt": "Halvardt Bautechnik (DE)"}
_conn = None


def conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = store.connect()
    return _conn


class Query(BaseModel):
    company: str
    question: str


@app.post("/ask")
def ask_endpoint(q: Query):
    if q.company not in COMPANIES:
        return {"error": "unknown company"}
    ans, hits, timing = ask(conn(), q.company, q.question)
    return {"answer": ans.text, "refused": ans.refused, "flags": ans.flags, "citations": ans.citations,
            "passages": [{"doc_id": h["doc_id"], "heading": h["heading"], "nature": h["nature"],
                          "score": round(h["score"], 3), "text": h["text"]} for h in hits[:5]],
            "timing": timing}


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Accounting RAG</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e3e2dc;--accent:#2f5d50;--warn:#9a5b00}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a3a29b;--card:#1f1f1d;--line:#33332f;--accent:#7fb8a6;--warn:#e0a54a}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:32px 16px}
h1{font-size:1.4rem;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 24px}
form{display:flex;flex-wrap:wrap;gap:8px}select,input,button{font:inherit;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}
input{flex:1 1 320px}button{background:var(--accent);color:#fff;border:0;cursor:pointer}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-top:20px;white-space:pre-wrap}
.warn{color:var(--warn)}small{color:var(--muted)}ul{padding-left:18px}li{margin:6px 0}
</style></head><body><main>
<h1>Ask the accounting documents</h1>
<p class="sub">Invented documents of three invented companies. You only ever see your own company's documents.</p>
<form id="f"><select id="c">%OPTIONS%</select><input id="q" placeholder="e.g. What hourly rate did we agree with Ashcombe Advisory Ltd?" required>
<button>Ask</button></form><div id="out"></div></main>
<script>
const out=document.getElementById('out');
document.getElementById('f').addEventListener('submit',async e=>{e.preventDefault();out.innerHTML='<div class="card">…</div>';
const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({company:c.value,question:q.value})});
const d=await r.json();const esc=s=>String(s).replace(/[&<>"]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]));
let h='<div class="card'+(d.flags&&d.flags.length?' warn':'')+'">'+esc(d.answer||d.error)+'</div>';
if(d.passages){h+='<div class="card"><small>Top passages · '+d.timing.retrieval_ms+' ms retrieval, '+d.timing.answer_ms+' ms answer</small><ul>'+
d.passages.map(p=>'<li><b>'+esc(p.doc_id)+' › '+esc(p.heading)+'</b> <small>'+esc(p.nature)+' · '+p.score+'</small><br>'+esc(p.text)+'</li>').join('')+'</ul></div>';}
out.innerHTML=h;});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.replace("%OPTIONS%", "".join(f'<option value="{k}">{v}</option>' for k, v in COMPANIES.items()))
