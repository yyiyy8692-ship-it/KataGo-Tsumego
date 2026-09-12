/* 答案模式：上传 → 轮询 → 逐题渲染 */
(function () {
  'use strict';
  var $ = function (s) { return document.querySelector(s); };
  var fileInput = null, polling = false;
  var state = { tid: new URLSearchParams(location.search).get('tid') || $('body').dataset.tid || '', after: 0, boards: {} };

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
    fetch('/api/solve', { method: 'POST', body: fd }).then(function (r) {
      return r.json().then(function (d) { return { ok: r.ok, d: d }; });
    }).then(function (res) {
      $('#loading').classList.add('hidden');
      if (!res.ok) { showError(res.d.error || '上传失败'); return; }
      state.tid = res.d.tid;
      state.after = 0; state.boards = {};
      $('#results').innerHTML = '';
      $('#err').classList.add('hidden');
      $('#progress-card').classList.remove('hidden');
      history.replaceState(null, '', '/solve/' + state.tid);
      startPoll();
    }).catch(function (e) {
      $('#loading').classList.add('hidden');
      showError('网络错误：' + e);
    });
  });

  function showError(msg) {
    $('#err').textContent = msg;
    $('#err').classList.remove('hidden');
  }

  /* ---------- 轮询 ---------- */
  function startPoll() {
    if (polling) return;
    polling = true;
    poll();
  }

  function poll() {
    fetch('/api/task/' + state.tid + '?after=' + state.after)
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d.error) { showError(d.error); polling = false; return; }
        render(d);
        if (d.status === 'running') setTimeout(poll, 1500);
        else polling = false;
      })
      .catch(function (e) {
        showError('连接中断，正在重试…（' + e + '）');
        setTimeout(poll, 3000);
      });
  }

  function render(d) {
    var pct = 0;
    if (d.status === 'done') pct = 100;
    else if (d.total) pct = Math.round(d.solved / d.total * 100);
    $('#prog-fill').style.width = pct + '%';
    $('#prog-text').textContent = (d.status === 'done')
      ? ('全部完成，共 ' + d.total + ' 题')
      : d.stage;
    $('#prog-count').textContent = d.total
      ? (d.solved + ' / ' + d.total + ' 题') : '';
    if (d.error) {
      $('#prog-text').textContent = '处理中断';
      showError(d.error);
    }

    (d.boards || []).forEach(function (b) {
      state.boards[b.idx] = b;
      if (b.status === 'done' || b.status === 'error') paint(b.idx);
    });
    // 只有前面连续做完的题才不再回传
    var i = 0;
    while (state.boards[i] &&
           ['done', 'error'].indexOf(state.boards[i].status) >= 0) i++;
    state.after = i;
  }

  function paint(idx) {
    var b = state.boards[idx];
    if ($('#b' + idx)) $('#b' + idx).remove();
    var box = document.createElement('section');
    box.className = 'pb' + (b.status === 'error' ? ' error' : '');
    box.id = 'b' + idx;

    if (b.status === 'error') {
      box.innerHTML = '<div class="pb-head"><h2>' + esc(b.label || ('第' + (idx + 1) + '题')) +
        '</h2></div><p class="meta">' + esc(b.error) + '</p>' +
        '<details class="orig"><summary>看原题</summary><img src="' + b.crop_url + '"></details>';
      $('#results').appendChild(box);
      return;
    }

    var tags = '';
    if (b.is_global) tags += '<span class="tag global">全局实战题</span>';
    else tags += '<span class="tag">手筋/死活题</span>';
    if (b.n_solutions > 1) {
      tags += '<span class="tag multi">' + b.n_solutions + ' 个正解</span>';
    }

    var html = '<div class="pb-head"><h2>' + esc(b.label) + '</h2>' + tags + '</div>';
    html += '<p class="meta">' + b.cols + '×' + b.rows + ' ｜ 黑 ' + b.n_black +
      ' 子 · 白 ' + b.n_white + ' 子 ｜ 起手黑领先 ' + Number(b.lead0).toFixed(1) + ' 目</p>';

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

    html += '<details class="orig"><summary>核对：识别出的题图</summary><img src="' +
      b.crop_url + '" alt="题图"></details>';
    box.innerHTML = html;
    $('#results').appendChild(box);
  }

  /* 刷新或直接打开分享链接时，续着上次的进度看 */
  if (state.tid) {
    $('#progress-card').classList.remove('hidden');
    $('#prog-text').textContent = '正在载入上次的结果…';
    startPoll();
  }
})();
