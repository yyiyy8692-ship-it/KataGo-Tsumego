/* 死活题批改前端：四步流程 */
const GC = "ABCDEFGHJKLMNOPQRST";
let SID = null, REC = null, PTYPE = null;
let stones = {};          // "x,y" -> "B"/"W"（确认用）
let kidMoves = [];        // [{seq,color,x,y}]
let digits = [];          // 识别到的手写编号

const $ = s => document.querySelector(s);
const show = id => { $(id).classList.remove("hidden"); };
const hide = id => { $(id).classList.add("hidden"); };
const loading = (on, text) => {
  $("#loading-text").textContent = text || "处理中…";
  on ? show("#loading") : hide("#loading");
};
const setStep = n => {
  document.querySelectorAll(".step-dot").forEach(d =>
    d.classList.toggle("active", +d.dataset.s <= n));
  [1,2,3,4].forEach(i => (i === n ? show("#step"+i) : hide("#step"+i)));
};

/* ---------- SVG 棋盘 ---------- */
function renderBoard(el, cols, rows, stoneMap, labels, onClick) {
  const cell = 44, margin = 34;
  const W = (cols-1)*cell + 2*margin, H = (rows-1)*cell + 2*margin;
  const px = (x,y) => [margin + x*cell, margin + y*cell];
  let s = `<svg viewBox="0 0 ${W} ${H}">`;
  s += `<rect width="${W}" height="${H}" fill="#E9C992" rx="8"/>`;
  for (let x=0; x<cols; x++) { const [a,b]=px(x,0),[c,d]=px(x,rows-1);
    s += `<line x1="${a}" y1="${b}" x2="${c}" y2="${d}" stroke="#5F4A2A"/>`;
    s += `<text x="${a}" y="${margin-10}" font-size="13" fill="#5F4A2A" text-anchor="middle">${GC[x]}</text>`; }
  for (let y=0; y<rows; y++) { const [a,b]=px(0,y),[c,d]=px(cols-1,y);
    s += `<line x1="${a}" y1="${b}" x2="${c}" y2="${d}" stroke="#5F4A2A"/>`;
    s += `<text x="${margin-16}" y="${b+4}" font-size="13" fill="#5F4A2A" text-anchor="middle">${rows-y}</text>`; }
  const r = cell*0.44;
  for (const k in stoneMap) {
    const [x,y] = k.split(",").map(Number), [cx,cy] = px(x,y);
    const c = stoneMap[k];
    s += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${c==="B"?"#26221C":"#F7F3EA"}" stroke="#5F4A2A"/>`;
  }
  for (const k in (labels||{})) {
    const [x,y] = k.split(",").map(Number), [cx,cy] = px(x,y);
    const {text, color} = labels[k];
    s += `<text x="${cx}" y="${cy+5}" font-size="15" font-weight="bold" fill="${color}" text-anchor="middle">${text}</text>`;
  }
  if (onClick) {
    for (let x=0; x<cols; x++) for (let y=0; y<rows; y++) {
      const [cx,cy] = px(x,y);
      s += `<rect x="${cx-cell/2}" y="${cy-cell/2}" width="${cell}" height="${cell}" fill="transparent" data-x="${x}" data-y="${y}" style="cursor:pointer"/>`;
    }
  }
  s += "</svg>";
  el.innerHTML = s;
  if (onClick) el.querySelectorAll("rect[data-x]").forEach(rc =>
    rc.addEventListener("click", () => onClick(+rc.dataset.x, +rc.dataset.y)));
}

/* ---------- 步骤1：上传 ---------- */
$("#photo").addEventListener("change", e => {
  const f = e.target.files[0];
  if (!f) return;
  const img = $("#preview");
  img.src = URL.createObjectURL(f);
  img.classList.remove("hidden");
  checkReady();
});
document.querySelectorAll("input[name=ptype]").forEach(r =>
  r.addEventListener("change", checkReady));
function checkReady() {
  $("#btn-upload").disabled = !($("#photo").files[0] &&
    document.querySelector("input[name=ptype]:checked"));
}
$("#btn-upload").addEventListener("click", async () => {
  const fd = new FormData();
  fd.append("photo", $("#photo").files[0]);
  fd.append("ptype", document.querySelector("input[name=ptype]:checked").value);
  fd.append("note", $("#note").value);
  loading(true, "识别棋形中…");
  const res = await fetch("/api/upload", {method:"POST", body:fd}).then(r=>r.json());
  loading(false);
  if (res.error) { alert(res.error); return; }
  SID = res.sid; REC = res.rec; PTYPE = fd.get("ptype");
  $("#overlay-img").src = res.overlay_url;
  stones = {};
  REC.black.forEach(([x,y]) => stones[x+","+y] = "B");
  REC.white.forEach(([x,y]) => stones[x+","+y] = "W");
  digits = REC.digits || [];
  drawStep2();
  setStep(2);
});

/* ---------- 步骤2：棋形确认 ---------- */
function drawStep2() {
  renderBoard($("#board2"), REC.cols, REC.rows, stones, digitLabels(), (x,y) => {
    const k = x+","+y;
    stones[k] = stones[k] === "B" ? "W" : stones[k] === "W" ? undefined : "B";
    if (stones[k] === undefined) delete stones[k];
    drawStep2();
  });
}
function digitLabels() {
  const L = {};
  digits.forEach(([seq,x,y]) => {
    L[x+","+y] = {text: seq || "?", color: "#2563eb"};
  });
  return L;
}
$("#btn-toggle-overlay").addEventListener("click", () =>
  $("#overlay-img").classList.toggle("hidden"));
$("#btn-confirm").addEventListener("click", async () => {
  const black = [], white = [];
  for (const k in stones) { const [x,y] = k.split(",").map(Number);
    (stones[k]==="B"?black:white).push([x,y]); }
  loading(true);
  await fetch("/api/confirm", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({sid:SID, black, white})});
  loading(false);
  kidMoves = [];
  drawStep3();
  setStep(3);
});

/* ---------- 步骤3：录答案 ---------- */
function toPlayFirst() { return PTYPE.startsWith("B") ? "B" : "W"; }
function drawStep3() {
  const L = digitLabels();
  kidMoves.forEach(m => { L[m.x+","+m.y] = {text:m.seq, color: m.color==="B"?"#F7F3EA":"#26221C"}; });
  const sm = {...stones};
  kidMoves.forEach(m => sm[m.x+","+m.y] = m.color);
  renderBoard($("#board3"), REC.cols, REC.rows, sm, L, (x,y) => {
    if (kidMoves.some(m => m.x===x && m.y===y)) return;
    if (stones[x+","+y]) { alert("这个点已有棋子，孩子只能下在空点上"); return; }
    const color = kidMoves.length % 2 === 0 ? toPlayFirst()
      : (toPlayFirst()==="B"?"W":"B");
    kidMoves.push({seq: kidMoves.length+1, color, x, y});
    drawStep3();
  });
  $("#moves-list").innerHTML = kidMoves.map(m =>
    `<span>${m.seq} ${m.color==="B"?"黑":"白"} ${GC[m.x]}${REC.rows-m.y}</span>`).join("")
    || "还没录入手顺";
}
$("#btn-undo").addEventListener("click", () => { kidMoves.pop(); drawStep3(); });
$("#btn-clear").addEventListener("click", () => { kidMoves = []; drawStep3(); });
$("#btn-grade").addEventListener("click", async () => {
  loading(true, "KataGo 计算中（逐手验证孩子变化线）…");
  const res = await fetch("/api/grade", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({sid:SID, kid_moves:kidMoves})}).then(r=>r.json());
  loading(false);
  if (res.error) { alert(res.error); return; }
  drawResult(res);
  setStep(4);
});

/* ---------- 步骤4：结果 ---------- */
function drawResult(res) {
  const R = res.result;
  $("#alerts").innerHTML = (R.alerts||[]).map(a => `<div class="alert">⚠️ ${a}</div>`).join("");
  let v = `<b>${R.type_desc}</b><br>正解首着：<b>${R.solution.move}</b>　结论：<b>${R.solution.verdict}</b><br>`;
  if (R.kid_report && R.kid_report.length) {
    v += "孩子变化线：<br>" + R.kid_report.map(m => {
      const cls = m.is_best ? "good" : (m.rank && m.rank<=3 ? "" : "bad");
      const tag = m.is_best ? "✓ 引擎首选" : (m.rank ? `第${m.rank}选（差${m.lead_gap}目）` : `✗ 引擎首选 ${m.engine_best}`);
      return `<span class="${cls}">${m.seq}@${m.move} ${tag}</span>`;
    }).join("<br>");
    v += `<br>变化线结局：<b>${R.kid_verdict || "未定"}</b>（正解结局 ${R.solution.verdict}）`;
  } else {
    v += "未录入孩子手顺，只给出正解。";
  }
  $("#verdict-box").innerHTML = v;
  $("#hints").innerHTML = res.hints.map(h => `<li>${h}</li>`).join("");
  // 正解图：孩子答案画在步骤4棋盘
  const L = {};
  kidMoves.forEach(m => { L[m.x+","+m.y] = {text:m.seq, color: m.color==="B"?"#F7F3EA":"#26221C"}; });
  const sm = {...stones};
  kidMoves.forEach(m => sm[m.x+","+m.y] = m.color);
  renderBoard($("#board4"), REC.cols, REC.rows, sm, L, null);
  $("#report-link").href = "/report/" + SID;
}
$("#btn-restart").addEventListener("click", () => location.reload());
