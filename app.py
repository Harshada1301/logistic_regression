"""
GenAI Study-Impact Predictor - Logistic Regression + Flask (single file).
Keep app.py, requirements.txt and logistic.pkl in the SAME folder.
Render start command:  gunicorn app:app
"""
import glob
import json
import os
import pickle

import pandas as pd
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE, "logistic.pkl")
if not os.path.exists(MODEL_FILE):  # Linux is case-sensitive -> fall back to any .pkl
    MODEL_FILE = (glob.glob(os.path.join(BASE, "*.pkl")) + glob.glob(os.path.join(BASE, "*.PKL")))[0]
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# Exact training column order
FEATURES = [
    "Major_Category", "Year_of_Study", "Pre_Semester_GPA", "Weekly_GenAI_Hours",
    "Primary_Use_Case", "Prompt_Engineering_Skill", "Paid_Subscription",
    "Traditional_Study_Hours", "Perceived_AI_Dependency", "Institutional_Policy",
    "Skill_Retention_Score",
]

# ---------------------------------------------------------------------------
# EDIT ME: the model stores categories as numbers. Make these labels / codes
# match the encoding you used in training  ->  "label shown": numeric code
# ---------------------------------------------------------------------------
OPTIONS = {
    "major": {"Arts & Humanities": 0, "Business": 1, "Engineering": 2, "Health Sciences": 3, "Science": 4},
    "use_case": {"Brainstorming": 0, "Coding": 1, "Research": 2, "Summarizing": 3, "Writing": 4},
    "skill": {"Beginner": 0, "Intermediate": 1, "Advanced": 2},
    "paid": {"No": 0, "Yes": 1},
    "policy": {"Banned": 0, "Restricted": 1, "Allowed": 2},
}

# EDIT ME: what each predicted class (0, 1, 2 ...) means in your project.
CLASS_LABELS = {"0": "Class 0", "1": "Class 1", "2": "Class 2"}


@app.route("/")
def home():
    page = (PAGE.replace("__OPTIONS__", json.dumps(OPTIONS))
                .replace("__LABELS__", json.dumps(CLASS_LABELS)))
    return Response(page, mimetype="text/html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json(force=True)
        vals = [
            float(d["major"]), float(d["year"]), float(d["gpa"]), float(d["genai_hours"]),
            float(d["use_case"]), float(d["skill"]), float(d["paid"]),
            float(d["study_hours"]), float(d["dependency"]), float(d["policy"]),
            float(d["retention"]),
        ]
        row = pd.DataFrame([vals], columns=FEATURES)
        pred = model.predict(row)[0]
        probs = model.predict_proba(row)[0]
        classes = [str(c) for c in model.classes_]
        return jsonify(
            ok=True,
            prediction=str(pred),
            classes=classes,
            probs=[round(float(p) * 100, 1) for p in probs],
        )
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 400


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NeuroStudy · GenAI Impact Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0618;--ink:#f3eeff;--mute:#9b90b8;--lime:#c6ff3d;--org:#ff8a3d;--pink:#ff3d9a;--vio:#8a5cff;--glass:rgba(255,255,255,.06);--line:rgba(255,255,255,.14)}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--ink);min-height:100vh;overflow-x:hidden;
background-image:radial-gradient(900px 600px at 0% 0%,rgba(138,92,255,.35),transparent 60%),radial-gradient(800px 600px at 100% 20%,rgba(255,61,154,.22),transparent 60%),radial-gradient(900px 600px at 50% 110%,rgba(198,255,61,.14),transparent 60%)}
#net{position:fixed;inset:0;z-index:0}
.wrap{position:relative;z-index:1;max-width:1120px;margin:0 auto;padding:42px 20px 60px}
header{text-align:center;margin-bottom:32px}
.pill{display:inline-flex;gap:9px;align-items:center;padding:8px 18px;border-radius:99px;border:1px solid var(--line);background:var(--glass);font-size:12px;letter-spacing:1.8px;text-transform:uppercase;color:var(--lime)}
.pill i{width:8px;height:8px;border-radius:50%;background:var(--lime);box-shadow:0 0 14px var(--lime);animation:bl 1.5s infinite}
@keyframes bl{50%{opacity:.2}}
h1{font-family:'Sora',sans-serif;font-size:clamp(34px,6.2vw,66px);line-height:1.06;margin:20px 0 12px;font-weight:800;letter-spacing:-1.5px}
h1 span{background:linear-gradient(100deg,var(--lime),var(--org),var(--pink),var(--vio),var(--lime));background-size:300% auto;-webkit-background-clip:text;background-clip:text;color:transparent;animation:sh 7s linear infinite}
@keyframes sh{to{background-position:300% center}}
header p{color:var(--mute);max-width:580px;margin:auto;line-height:1.65;font-weight:300}
.grid{display:grid;grid-template-columns:1.5fr .95fr;gap:24px;align-items:start}
@media(max-width:920px){.grid{grid-template-columns:1fr}}
.card{background:var(--glass);border:1px solid var(--line);border-radius:28px;padding:28px;backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);box-shadow:0 30px 80px rgba(0,0,0,.55),inset 0 1px 0 rgba(255,255,255,.12);position:relative;animation:up .9s cubic-bezier(.2,.8,.2,1) both}
.card.r{animation-delay:.15s}
@keyframes up{from{opacity:0;transform:translateY(40px)}to{opacity:1;transform:none}}
.card::before{content:"";position:absolute;inset:-1px;border-radius:28px;padding:1px;pointer-events:none;background:conic-gradient(from var(--a,0deg),transparent 0 72%,var(--lime),var(--pink),transparent);-webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude;animation:sp 6s linear infinite;opacity:.85}
@property --a{syntax:'<angle>';initial-value:0deg;inherits:false}
@keyframes sp{to{--a:360deg}}
.sec{font-family:'Sora';font-size:12px;letter-spacing:2.2px;text-transform:uppercase;color:var(--org);margin:0 0 14px;display:flex;align-items:center;gap:10px}
.sec::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line),transparent)}
.sec:not(:first-child){margin-top:26px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:16px 18px}
@media(max-width:560px){.row{grid-template-columns:1fr}}
.f{display:flex;flex-direction:column;gap:8px}.f.full{grid-column:1/-1}
label{font-size:11.5px;letter-spacing:1px;text-transform:uppercase;color:var(--mute);font-weight:500;display:flex;justify-content:space-between;align-items:baseline}
label b{font-family:'Sora';font-size:15px;letter-spacing:0;color:var(--lime);text-transform:none}
select{width:100%;padding:13px 38px 13px 14px;border-radius:13px;border:1px solid var(--line);background:rgba(255,255,255,.07);color:var(--ink);font:inherit;font-size:15px;outline:none;transition:.3s;appearance:none;-webkit-appearance:none;
background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%239b90b8' fill='none' stroke-width='2'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 14px center}
select option{color:#111}
select:focus{border-color:var(--lime);box-shadow:0 0 0 4px rgba(198,255,61,.16);transform:translateY(-2px)}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:6px;border-radius:9px;outline:none;cursor:pointer;background:linear-gradient(90deg,var(--lime),var(--org) var(--p,50%),rgba(255,255,255,.13) var(--p,50%))}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:22px;height:22px;border-radius:50%;background:#fff;border:4px solid var(--org);box-shadow:0 0 18px var(--org);transition:.2s}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.28)}
input[type=range]::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:#fff;border:4px solid var(--org)}
.btn{margin-top:28px;width:100%;padding:18px;border:0;border-radius:16px;cursor:pointer;position:relative;overflow:hidden;font:inherit;font-weight:700;font-size:16px;letter-spacing:1px;color:#150a00;background:linear-gradient(100deg,var(--lime),var(--org),var(--pink),var(--lime));background-size:300% auto;box-shadow:0 14px 36px rgba(255,138,61,.45);transition:.4s;animation:sh 6s linear infinite}
.btn:hover{transform:translateY(-3px);box-shadow:0 20px 46px rgba(255,61,154,.5)}
.btn::after{content:"";position:absolute;top:0;left:-70%;width:45%;height:100%;background:linear-gradient(100deg,transparent,rgba(255,255,255,.6),transparent);transform:skewX(-20deg);animation:sw 3s infinite}
@keyframes sw{to{left:150%}}
.btn.busy{pointer-events:none;opacity:.75}
.res{position:sticky;top:22px;text-align:center;min-height:440px;display:flex;flex-direction:column;align-items:center;justify-content:center}
.orb{position:relative;width:170px;height:170px;margin-bottom:18px;display:grid;place-items:center}
.orb::before,.orb::after{content:"";position:absolute;inset:0;border-radius:50%;border:2px solid transparent;border-top-color:var(--lime);border-right-color:var(--pink);animation:rot 3s linear infinite}
.orb::after{inset:14px;border-top-color:var(--org);border-right-color:var(--vio);animation-duration:2s;animation-direction:reverse}
@keyframes rot{to{transform:rotate(360deg)}}
.orb span{font-size:58px;animation:bob 3s ease-in-out infinite}
@keyframes bob{50%{transform:translateY(-7px) scale(1.06)}}
.tag{font-size:11.5px;letter-spacing:2px;text-transform:uppercase;color:var(--mute)}
.verdict{font-family:'Sora';font-weight:800;font-size:clamp(28px,4.6vw,40px);margin:8px 0;background:linear-gradient(100deg,var(--lime),var(--org));-webkit-background-clip:text;background-clip:text;color:transparent}
.hint{color:var(--mute);font-size:14px;line-height:1.55;max-width:250px;font-weight:300}
.bars{width:100%;margin-top:20px;display:flex;flex-direction:column;gap:12px;text-align:left}
.bar .t{display:flex;justify-content:space-between;font-size:12.5px;margin-bottom:6px;color:var(--ink)}
.bar .t b{font-family:'Sora'}
.bar .tr{height:9px;border-radius:9px;background:rgba(255,255,255,.1);overflow:hidden}
.bar .fl{height:100%;width:0;border-radius:9px;background:linear-gradient(90deg,var(--vio),var(--pink));transition:width 1.2s cubic-bezier(.2,.8,.2,1)}
.bar.top .fl{background:linear-gradient(90deg,var(--lime),var(--org));box-shadow:0 0 14px rgba(198,255,61,.6)}
.pop{animation:pp .7s cubic-bezier(.2,1.4,.4,1)}
@keyframes pp{from{opacity:0;transform:scale(.75)}to{opacity:1;transform:none}}
.err{background:rgba(255,61,154,.12);border:1px solid rgba(255,61,154,.45);color:#ffd3e8;padding:12px 14px;border-radius:12px;font-size:14px;margin-top:14px}
footer{text-align:center;color:var(--mute);font-size:12px;margin-top:36px;opacity:.7}
</style>
</head>
<body>
<canvas id="net"></canvas>
<div class="wrap">
<header>
  <span class="pill"><i></i>Logistic Regression · Live Model</span>
  <h1>Predict the <span>GenAI impact</span> on your studies</h1>
  <p>Describe a student's study habits and AI usage — the model estimates the outcome class with a full probability breakdown.</p>
</header>
<div class="grid">
<section class="card">
<form id="form" autocomplete="off">
  <div class="sec">Academic profile</div>
  <div class="row">
    <div class="f"><label>Major category</label><select id="major"></select></div>
    <div class="f"><label>Year of study <b id="o_year">2</b></label><input type="range" id="year" min="1" max="6" value="2"></div>
    <div class="f full"><label>Pre-semester GPA <b id="o_gpa">3.00</b></label><input type="range" id="gpa" min="0" max="4" step="0.01" value="3"></div>
    <div class="f"><label>Traditional study hrs / week <b id="o_sh">12</b></label><input type="range" id="study_hours" min="0" max="60" step="0.5" value="12"></div>
    <div class="f"><label>Skill retention score <b id="o_rt">70</b></label><input type="range" id="retention" min="0" max="100" value="70"></div>
  </div>
  <div class="sec">GenAI usage</div>
  <div class="row">
    <div class="f"><label>Weekly GenAI hours <b id="o_gh">8</b></label><input type="range" id="genai_hours" min="0" max="50" step="0.5" value="8"></div>
    <div class="f"><label>Perceived AI dependency <b id="o_dp">5</b></label><input type="range" id="dependency" min="0" max="10" step="0.5" value="5"></div>
    <div class="f"><label>Primary use case</label><select id="use_case"></select></div>
    <div class="f"><label>Prompt engineering skill</label><select id="skill"></select></div>
    <div class="f"><label>Paid subscription</label><select id="paid"></select></div>
    <div class="f"><label>Institutional policy</label><select id="policy"></select></div>
  </div>
  <button class="btn" id="btn" type="submit">🧠 Predict outcome</button>
</form>
</section>

<aside class="card r res" id="res">
  <div class="orb"><span id="emoji">🧠</span></div>
  <div class="tag">Predicted outcome</div>
  <div class="verdict" id="verdict">Awaiting input</div>
  <p class="hint" id="hint">Set the sliders and press <b>Predict outcome</b>.</p>
  <div class="bars" id="bars"></div>
</aside>
</div>
<footer>Built with Flask &amp; scikit-learn · Predictions are model outputs, not guarantees</footer>
</div>

<script>
const OPT=__OPTIONS__, LABELS=__LABELS__, $=id=>document.getElementById(id);
for(const k in OPT){$(k).innerHTML=Object.entries(OPT[k]).map(([n,v])=>`<option value="${v}">${n}</option>`).join('')}
const outs={year:'o_year',gpa:'o_gpa',study_hours:'o_sh',retention:'o_rt',genai_hours:'o_gh',dependency:'o_dp'};
function paint(el){el.style.setProperty('--p',(el.value-el.min)/(el.max-el.min)*100+'%');$(outs[el.id]).textContent=el.id==='gpa'?Number(el.value).toFixed(2):el.value}
document.querySelectorAll('input[type=range]').forEach(el=>{paint(el);el.addEventListener('input',()=>paint(el))});

/* neural-network background */
(function(){const c=$('net'),x=c.getContext('2d');let w,h,n=[],m={x:-999,y:-999};
function init(){w=c.width=innerWidth;h=c.height=innerHeight;n=Array.from({length:Math.min(70,Math.floor(w*h/22000))},()=>({x:Math.random()*w,y:Math.random()*h,vx:(Math.random()-.5)*.4,vy:(Math.random()-.5)*.4}))}
init();addEventListener('resize',init);addEventListener('mousemove',e=>{m.x=e.clientX;m.y=e.clientY});
(function loop(){x.clearRect(0,0,w,h);
n.forEach(p=>{p.x+=p.vx;p.y+=p.vy;if(p.x<0||p.x>w)p.vx*=-1;if(p.y<0||p.y>h)p.vy*=-1});
for(let i=0;i<n.length;i++){for(let j=i+1;j<n.length;j++){const d=Math.hypot(n[i].x-n[j].x,n[i].y-n[j].y);
 if(d<150){x.strokeStyle=`rgba(198,255,61,${.16*(1-d/150)})`;x.beginPath();x.moveTo(n[i].x,n[i].y);x.lineTo(n[j].x,n[j].y);x.stroke()}}
 const dm=Math.hypot(n[i].x-m.x,n[i].y-m.y);if(dm<190){x.strokeStyle=`rgba(255,61,154,${.45*(1-dm/190)})`;x.beginPath();x.moveTo(n[i].x,n[i].y);x.lineTo(m.x,m.y);x.stroke()}
 x.fillStyle='rgba(255,170,90,.75)';x.beginPath();x.arc(n[i].x,n[i].y,2,0,6.3);x.fill()}
requestAnimationFrame(loop)})()})();

const EMO=['🌱','⚡','🚀','🔥','💎'];
$('form').addEventListener('submit',async e=>{
  e.preventDefault();const b=$('btn');b.classList.add('busy');b.textContent='⏳ Analysing...';
  const old=document.querySelector('.err');if(old)old.remove();
  const body={};['major','year','gpa','genai_hours','use_case','skill','paid','study_hours','dependency','policy','retention'].forEach(k=>body[k]=$(k).value);
  try{
    const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();if(!d.ok)throw new Error(d.error);
    const name=c=>LABELS[c]||('Class '+c), top=d.probs.indexOf(Math.max(...d.probs));
    const v=$('verdict');v.textContent=name(d.prediction);v.classList.remove('pop');void v.offsetWidth;v.classList.add('pop');
    $('emoji').textContent=EMO[top%EMO.length];
    $('hint').textContent='Highest probability class, with the full breakdown below.';
    $('bars').innerHTML=d.classes.map((c,i)=>`<div class="bar ${i===top?'top':''}"><div class="t"><span>${name(c)}</span><b>${d.probs[i]}%</b></div><div class="tr"><div class="fl"></div></div></div>`).join('');
    setTimeout(()=>document.querySelectorAll('.fl').forEach((f,i)=>f.style.width=d.probs[i]+'%'),40);
    if(innerWidth<920)$('res').scrollIntoView({behavior:'smooth',block:'center'});
  }catch(err){$('res').insertAdjacentHTML('beforeend',`<div class="err">⚠️ ${err.message}</div>`)}
  b.classList.remove('busy');b.textContent='🧠 Predict outcome';
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
