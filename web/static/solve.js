/* 答案模式：上传 → 识别 → **人工逐题确认** → 点哪题算哪题 */
(function () {
  'use strict';
  var $ = function (s) { return document.querySelector(s); };
  var fileInput = null, polling = false;
  var state = {
    tid: new URLSearchParams(location.search).get('tid') || $('body').dataset.tid || '',
    boards: {},   // idx -> 服务端给的题数据
    got: {},      // idx -> 1，答案已经拿到（告诉服务端别再回传）
    skipped: {}   // idx -> 1，用户确认时剔掉的题
  };

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }

  /* ---------- 选择文件 ---------- */
  document.querySelectorAll('.entry input').forEach(function (inp) {
    inp.addEventListener('change', function () {
      if (!inp.files || !inp.files[0]) return;
      fileInput = inp.files[0];
      if (inp.dataset.kind === 'pdf') {
        $('#preview').classList.add('hidden');
      } else {
        $('#preview').src = URL.createObjectURL(fileInput);
        $('#preview').classList.remove('hidden');
      }
      $('#preview-box').classList.remove('hidden');
      $('#upload-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });

  $('#cancel').addEventListener('click', function () {
    fileInput = null;
    document.querySelectorAll('.entry input').forEach(function (i) { i.value = ''; });
    $('#preview-box').classList.add('hidden');
  });

  $('#go').addEventListener('click', function () {
    if (!fileInput) return;
    var fd = new FormData();
    fd.append('file', fileInput);
    fd.append('visits', $('#visits').value);
    fd.append('depth', $('#depth').value);
    fd.append('tol', $('#tol').value);
    $('#loading').classList.remove('hidden');
    $('#loading-text').textContent = '上传中…';
    fetch('/api/detect', { method: 'POST', body: fd }).then(function (r) {
      return r.json().then(function (d) { return { ok: r.ok, d: d }; });
    }).then(function (res) {
      $('#loading').classList.add('hidden');
      if (!res.ok) { showError(res.d.error || '上传失败'); return; }
      reset(res.d.tid);
    }).catch(function (e) {
      $('#loading').classList.add('hidden');
      showError('网络错误：' + e);
    });
  });

  function reset(tid) {
    state.tid = tid; state.boards = {}; state.got = {}; state.skipped = {};
    $('#results').innerHTML = '';
    $('#err').classList.add('hidden');
    $('#progress-card').classList.remove('hidden');
    $('#confirm-bar').classList.add('hidden');
    history.replaceState(null, '', '/solve/' + tid);
    startPoll();
  }

  function showError(msg) {
    $('#err').textContent = msg;
    $('#err').classList.remove('hidden');
  }

  /* ---------- 轮询 ---------- */
  function startPoll() { if (!polling) { polling = true; poll(); } }

  function poll() {
    var have = Object.keys(state.got).join(',');
    fetch('/api/task/' + state.tid + '?have=' + have)
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d.error) { showError(d.error); polling = false; return; }
        render(d);
        if (busy(d)) setTimeout(poll, 1500);
        else polling = false;
      })
      .catch(function (e) {
        showError('连接中断，正在重试…（' + e + '）');
        setTimeout(poll, 3000);
      });
  }

  /* 还有活儿在干才继续轮询：识别中、或有题在排队/计算中 */
  function busy(d) {
    if (d.status === 'running') return true;
    if (d.pending > 0) return true;
    return (d.boards || []).some(function (b) {
      return b.status === 'queued' || b.status === 'running';
    });
  }

  function render(d) {
    var doneN = Object.keys(state.got).length;
    $('#prog-text').textContent = d.status === 'running'
      ? d.stage
      : (d.pending ? d.stage
        : (d.total ? '识别到 ' + d.total + ' 题' + (doneN ? '，已算出 ' + doneN + ' 题' : '')
          : d.stage));
    $('#prog-count').textContent = d.total
      ? (doneN + ' / ' + d.total + ' 题已算') : '';
    $('#prog-fill').style.width =
      (d.total ? Math.round(doneN / d.total * 100) : 0) + '%';
    if (d.error) { $('#prog-text').textContent = '处理中断'; showError(d.error); }

    (d.boards || []).forEach(function (b) {
      var prev = state.boards[b.idx];
      if (prev && prev.solutions && !b.solutions) b.solutions = prev.solutions;
      if (prev && prev.check_svg && !b.check_svg) b.check_svg = prev.check_svg;
      state.boards[b.idx] = b;
      if (b.solutions) state.got[b.idx] = 1;
      var changed = !prev || prev.status !== b.status ||
        (prev.solutions ? false : !!b.solutions);
      if (changed) paint(b.idx);
    });

    // 识别完成且还有题没点，就亮出「全部计算」
    var waitN = Object.keys(state.boards).filter(function (i) {
      return state.boards[i].status === 'detected' && !state.skipped[i];
    }).length;
    if (d.status === 'ready' && waitN > 0) {
      $('#confirm-bar').classList.remove('hidden');
      $('#ready-text').textContent = '共 ' + d.total + ' 题，还有 ' + waitN +
        ' 题待确认';
      $('#solve-all').disabled = false;
      $('#solve-all').textContent = '全部计算（' + waitN + ' 题）';
    } else if (d.status === 'ready') {
      $('#confirm-bar').classList.remove('hidden');
      $('#ready-text').textContent = '共 ' + d.total + ' 题，已全部处理';
      $('#solve-all').disabled = true;
      $('#solve-all').textContent = '全部计算';
    }
    (d.notes || []).forEach(function (n) {
      if (!$('#err').classList.contains('hidden')) return;
    });
  }

  function paint(idx) {
    var b = state.boards[idx];
    if (!b) return;
    var old = $('#b' + idx);
    if (old) old.remove();
    if (state.skipped[idx]) return;

    var box = document.createElement('section');
    box.className = 'pb' + (b.status === 'error' ? ' error' : '');
    box.id = 'b' + idx;
    var title = b.label || ('第' + (idx + 1) + '题');

    if (b.status === 'error') {
      box.innerHTML = '<div class="pb-head"><h2>' + esc(title) + '</h2></div>' +
        '<p class="meta">' + esc(b.error || '识别失败') + '</p>' +
        origHtml(b);
      $('#results').appendChild(box);
      return;
    }

    var tags = '';
    tags += b.is_global ? '<span class="tag global">全局实战题</span>'
      : '<span class="tag">手筋/死活题</span>';
    if (b.status === 'queued' || b.status === 'running') {
      tags += '<span class="tag wait">' +
        (b.status === 'queued' ? '排队中' : '计算中') + '</span>';
    } else if (b.status === 'done' && b.n_solutions > 1) {
      tags += '<span class="tag multi">' + b.n_solutions + ' 个正解</span>';
    } else if (b.status === 'done') {
      tags += '<span class="tag ok">已算出</span>';
    }

    var html = '<div class="pb-head"><h2>' + esc(title) + '</h2>' + tags + '</div>';
    html += '<p class="meta">' + b.cols + '×' + b.rows + ' ｜ 黑 ' + b.n_black +
      ' 子 · 白 ' + b.n_white + ' 子' +
      (b.lead0 !== undefined && b.lead0 !== null
        ? ' ｜ 起手黑领先 ' + Number(b.lead0).toFixed(1) + ' 目' : '') + '</p>';

    if (b.status === 'detected') {
      html += '<p class="tip">先核对棋形：下面是识别出来的样子。</p>';
      html += '<div class="board-wrap">' + b.check_svg + '</div>';
      html += origHtml(b);
      html += '<div class="act">' +
        '<button class="primary" data-solve="' + idx + '">✓ 棋形对了，算答案</button>' +
        '<button class="ghost" data-skip="' + idx + '">这题不要</button></div>';
      box.innerHTML = html;
      $('#results').appendChild(box);
      return;
    }

    if (b.status === 'queued' || b.status === 'running') {
      html += '<div class="board-wrap">' + (b.check_svg || '') + '</div>';
      html += '<div class="waitline"><span class="spinner small"></span>' +
        (b.status === 'queued' ? '排队中，前面的题算完就轮到它…' :
          'KataGo 计算中，约 30~60 秒…') + '</div>';
      box.innerHTML = html;
      $('#results').appendChild(box);
      return;
    }

    // done
    (b.solutions || []).forEach(function (s) {
      html += '<div class="sol"><p class="lead-line">' + esc(s.title) + '</p>';
      html += '<div class="board-wrap">' + s.svg + '</div>';
      html += '<div class="explain">' + s.explain.map(function (t) {
        return '<div>' + esc(t) + '</div>';
      }).join('') + '</div>';
      if (s.alerts && s.alerts.length) {
        html += '<div class="alerts">' + s.alerts.map(function (a) {
          return '<div>' + esc(a) + '</div>';
        }).join('') + '</div>';
      }
      html += '</div>';
    });
    html += origHtml(b);
    box.innerHTML = html;
    $('#results').appendChild(box);
  }

  function origHtml(b) {
    return '<details class="orig"><summary>和拍的原图对照</summary><img src="' +
      b.crop_url + '" alt="题图"></details>';
  }

  /* ---------- 点「算答案」/「这题不要」 ---------- */
  $('#results').addEventListener('click', function (e) {
    var t = e.target;
    if (t.dataset && t.dataset.solve !== undefined) {
      var idx = parseInt(t.dataset.solve, 10);
      t.disabled = true;
      t.textContent = '已加入队列…';
      fetch('/api/board/' + state.tid + '/' + idx, { method: 'POST' })
        .then(function (r) { return r.json(); })
        .then(function (d) {
          if (d.error) { showError(d.error); return; }
          if (state.boards[idx]) state.boards[idx].status = 'queued';
          startPoll();
        })
        .catch(function (err) { showError('请求失败：' + err); });
    } else if (t.dataset && t.dataset.skip !== undefined) {
      var i2 = parseInt(t.dataset.skip, 10);
      state.skipped[i2] = 1;
      var box = $('#b' + i2);
      if (box) box.remove();
    }
  });

  $('#solve-all').addEventListener('click', function () {
    var want = Object.keys(state.boards).filter(function (i) {
      return state.boards[i].status === 'detected' && !state.skipped[i];
    }).map(Number);
    if (!want.length) return;
    $('#solve-all').disabled = true;
    $('#solve-all').textContent = '已加入队列…';
    fetch('/api/board/' + state.tid + '/all', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ idx: want })
    }).then(function (r) { return r.json(); })
      .then(function () { startPoll(); })
      .catch(function (e) { showError('请求失败：' + e); });
  });

  /* 刷新或直接打开分享链接时，续着上次的进度看 */
  if (state.tid) {
    $('#progress-card').classList.remove('hidden');
    $('#prog-text').textContent = '正在载入上次的结果…';
    startPoll();
  }
})();
