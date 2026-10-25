// Pure-JS inference for the small CNN-BiLSTM-Attention demo model (weights exported from PyTorch).
(function (root) {
  const sig = x => 1 / (1 + Math.exp(-x));
  function matvec(W, x, b) { // W: rows x cols (array of arrays)
    const out = new Array(W.length);
    for (let i = 0; i < W.length; i++) { let s = b ? b[i] : 0; const r = W[i]; for (let j = 0; j < x.length; j++) s += r[j] * x[j]; out[i] = s; }
    return out;
  }
  function lstmDir(seq, p) { // PyTorch gate order: i, f, g, o
    const H = p.w_hh[0].length; let h = new Array(H).fill(0), c = new Array(H).fill(0); const outs = [];
    for (const x of seq) {
      const a = matvec(p.w_ih, x, p.b_ih), r = matvec(p.w_hh, h, p.b_hh);
      const nh = new Array(H), nc = new Array(H);
      for (let k = 0; k < H; k++) {
        const i = sig(a[k] + r[k]), f = sig(a[H + k] + r[H + k]);
        const g = Math.tanh(a[2 * H + k] + r[2 * H + k]), o = sig(a[3 * H + k] + r[3 * H + k]);
        nc[k] = f * c[k] + i * g; nh[k] = o * Math.tanh(nc[k]);
      }
      h = nh; c = nc; outs.push(h);
    }
    return outs;
  }
  function forwardScaled(m, x) { // x: scaled window (length T)
    const F = m.conv_w.length, K = m.conv_w[0][0].length, T = x.length, z = [];
    for (let t = 0; t + K <= T; t++) {
      const v = new Array(F);
      for (let f = 0; f < F; f++) { let s = m.conv_b[f]; for (let k = 0; k < K; k++) s += m.conv_w[f][0][k] * x[t + k]; v[f] = Math.max(0, s); }
      z.push(v);
    }
    const fw = lstmDir(z, m.lstm_fw), bw = lstmDir(z.slice().reverse(), m.lstm_bw).reverse();
    const Hs = fw.map((h, t) => h.concat(bw[t]));
    const e = Hs.map(h => matvec(m.attn_v, matvec(m.attn_W, h, m.attn_b).map(Math.tanh))[0]);
    const mx = Math.max(...e), ex = e.map(v => Math.exp(v - mx)), S = ex.reduce((a, b) => a + b, 0);
    const alpha = ex.map(v => v / S), ctx = new Array(Hs[0].length).fill(0);
    Hs.forEach((h, t) => h.forEach((v, j) => { ctx[j] += alpha[t] * v; }));
    return { y: matvec(m.head_w, ctx, m.head_b)[0], alpha };
  }
  function predict(m, windowMs) { // window in m/s -> next-hour forecast in m/s
    const x = windowMs.map(v => (v - m.scaler_min) / m.scaler_range);
    const r = forwardScaled(m, x);
    const ys = m.target === "delta" ? r.y + x[x.length - 1] : r.y; // delta model: last reading + predicted change
    return { y: ys * m.scaler_range + m.scaler_min, alpha: r.alpha };
  }
  const api = { predict, forwardScaled };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.WindModel = api;
})(typeof self !== "undefined" ? self : this);
