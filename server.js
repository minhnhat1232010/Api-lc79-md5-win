/**
 * API DỰ ĐOÁN TÀI/XỈU - AI TỔNG HỢP (10 model, không random)
 * Nguồn dữ liệu: https://wtxmd52.tele68.com/v1/txmd5/sessions
 * Tác giả: Bạn
 * ID dấu nước: Tele@HoVanThien_Pro
 */

require('dotenv').config();
const express = require('express');
const axios = require('axios');
const fs = require('fs');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const path = require('path');

const APP_PORT = process.env.PORT || 3000;
const API_KEY_REQUIRED = !!process.env.API_KEY;

const STATE_PATH = path.join(__dirname, 'state.json');
function loadState() {
  try {
    const raw = fs.readFileSync(STATE_PATH, 'utf8');
    const obj = JSON.parse(raw);
    if (!Array.isArray(obj.pattern)) obj.pattern = [];
    return obj;
  } catch {
    return { pattern: [], lastUpdated: 0 };
  }
}
function saveState(state) {
  try {
    fs.writeFileSync(STATE_PATH, JSON.stringify(state, null, 2), 'utf8');
  } catch (e) {
    console.error('Lỗi ghi state.json:', e.message);
  }
}

const app = express();
app.use(helmet());
app.use(cors());
app.use(express.json());
app.use(morgan('combined'));

// middleware kiểm tra API key (tuỳ chọn)
app.use((req, res, next) => {
  if (!API_KEY_REQUIRED) return next();
  const key = req.headers['x-api-key'];
  if (key && key === process.env.API_KEY) return next();
  return res.status(401).json({ error: 'Unauthorized. Missing or invalid x-api-key.' });
});

const SOURCE_URL = 'https://wtxmd52.tele68.com/v1/txmd5/sessions';

// Utils
const toTX = (resultTruyenThong) => {
  // Chuẩn hoá về "T" | "X"
  if (!resultTruyenThong) return null;
  const s = String(resultTruyenThong).trim().toUpperCase();
  if (s === 'TAI' || s === 'TÀI' || s === 'T') return 'T';
  if (s === 'XIU' || s === 'XỈU' || s === 'X') return 'X';
  return null;
};
const toDisplay = (tOrx) => (tOrx === 'T' ? 'Tài' : 'Xỉu');
const clamp01 = (x) => Math.max(0, Math.min(1, x));

/**
 * Tính streak cuối cùng (ví dụ "TTTXX" -> streak cuối = 2 và ký tự "X")
 */
function lastStreak(arrTX) {
  if (arrTX.length === 0) return { char: null, len: 0 };
  let len = 1;
  for (let i = arrTX.length - 2; i >= 0; i--) {
    if (arrTX[i] === arrTX[i + 1]) len++;
    else break;
  }
  return { char: arrTX[arrTX.length - 1], len };
}

/**
 * Xây Markov 1 bậc từ pattern (P(next|current))
 */
function markov1(pattern) {
  const trans = { T: { T: 0, X: 0 }, X: { T: 0, X: 0 } };
  for (let i = 0; i < pattern.length - 1; i++) {
    const a = pattern[i], b = pattern[i + 1];
    if ((a === 'T' || a === 'X') && (b === 'T' || b === 'X')) {
      trans[a][b]++;
    }
  }
  const probs = { T: 0.5, X: 0.5 };
  const last = pattern[pattern.length - 1];
  if (last === 'T' || last === 'X') {
    const total = trans[last].T + trans[last].X;
    if (total > 0) {
      probs.T = trans[last].T / total;
      probs.X = trans[last].X / total;
    }
  }
  return { trans, probs };
}

/**
 * Tìm n-gram (k=3) gần nhất khớp đuôi pattern và xem "next" xảy ra gì trong lịch sử
 */
function ngramNext(pattern, k = 3) {
  if (pattern.length <= k) return { T: 0, X: 0 };
  const tail = pattern.slice(-k).join('');
  const counts = { T: 0, X: 0 };
  for (let i = 0; i < pattern.length - k; i++) {
    const gram = pattern.slice(i, i + k).join('');
    const nxt = pattern[i + k];
    if (gram === tail && (nxt === 'T' || nxt === 'X')) counts[nxt]++;
  }
  return counts;
}

/**
 * Điểm luân phiên (1-1)
 */
function alternationScore(pattern) {
  if (pattern.length < 3) return 0;
  let alt = 0, possible = 0;
  for (let i = 2; i < pattern.length; i++) {
    possible++;
    if (pattern[i] !== pattern[i - 1] && pattern[i - 1] !== pattern[i - 2] && pattern[i] === pattern[i - 2]) {
      alt++;
    }
  }
  return possible > 0 ? alt / possible : 0;
}

/**
 * Phân bố run-length (độ dài chuỗi liên tiếp T hoặc X)
 */
function runLengthDistribution(pattern) {
  const runs = [];
  let cur = pattern[0], len = 1;
  for (let i = 1; i < pattern.length; i++) {
    if (pattern[i] === cur) len++;
    else {
      runs.push({ char: cur, len });
      cur = pattern[i]; len = 1;
    }
  }
  if (cur) runs.push({ char: cur, len });
  return runs;
}

/**
 * Từ pattern -> xác suất Tai (pT) và Xiu (pX) theo 10 model (ensemble)
 */
function ensembleModels(pattern, recentTotals = []) {
  // pattern là mảng ['T','X',...], tối đa 20
  const n = pattern.length;
  const counts = { T: pattern.filter(x => x === 'T').length, X: pattern.filter(x => x === 'X').length };
  const pFreqT = n > 0 ? counts.T / n : 0.5;

  // Model 1: Đa số gần nhất (frequency)
  const m1 = pFreqT;

  // Model 2: Markov 1 bậc
  const mk = markov1(pattern);
  const m2 = mk.probs.T;

  // Model 3: Streak logic (nếu streak dài => xu hướng tiếp diễn, nếu vừa dài -> đảo chiều)
  const st = lastStreak(pattern);
  let m3 = 0.5;
  if (st.char) {
    // Nếu streak <=2 ưu tiên đảo chiều nhẹ, >3 ưu tiên tiếp diễn nhẹ
    if (st.len <= 2) m3 = st.char === 'T' ? 0.35 : 0.65;
    else if (st.len === 3) m3 = st.char === 'T' ? 0.55 : 0.45;
    else m3 = st.char === 'T' ? 0.60 : 0.40;
  }

  // Model 4: Alternation (1-1) detector
  const alt = alternationScore(pattern);
  // alt cao => nghiêng về đảo chiều
  const m4 = 0.5 + (pattern[n - 1] === 'T' ? -0.3 * alt : 0.3 * alt);

  // Model 5: N-gram (k=3)
  const ng = ngramNext(pattern, 3);
  const totalNG = ng.T + ng.X;
  const m5 = totalNG > 0 ? ng.T / totalNG : 0.5;

  // Model 6: Momentum 5 phiên gần nhất
  const k = Math.min(5, n);
  const lastK = pattern.slice(n - k);
  const cntK = { T: lastK.filter(x => x === 'T').length, X: lastK.filter(x => x === 'X').length };
  const m6 = k > 0 ? cntK.T / k : 0.5;

  // Model 7: Run-length distribution (nếu run dài thường xuất hiện nhiều, ưu tiên tiếp diễn)
  const runs = runLengthDistribution(pattern);
  const avgRun = runs.length ? runs.reduce((s, r) => s + r.len, 0) / runs.length : 1;
  const m7 = st.char ? (st.char === 'T' ? clamp01(0.45 + 0.05 * (st.len / avgRun)) : clamp01(0.55 - 0.05 * (st.len / avgRun))) : 0.5;

  // Model 8: Entropy / sự cân bằng (càng lệch -> tự tin hơn)
  const balance = Math.abs(pFreqT - 0.5); // 0..0.5
  const m8 = clamp01(0.5 + (pFreqT - 0.5) * (0.8 + balance)); // đẩy nhẹ theo lệch tổng quan

  // Model 9: Tổng điểm (nếu có) – thiên hướng Tài nếu tổng gần đây cao
  // recentTotals: mảng số (point). Nếu không có, giữ 0.5
  let m9 = 0.5;
  if (recentTotals.length > 0) {
    const avgTotal = recentTotals.reduce((s, x) => s + x, 0) / recentTotals.length; // 3-18
    // Chuẩn hoá về 0..1: 3 -> 0, 18 -> 1
    const norm = clamp01((avgTotal - 3) / 15);
    // Tài ~ điểm cao -> nghiêng theo norm
    m9 = 0.4 + 0.4 * norm; // 0.4..0.8
  }

  // Model 10: Pattern đảo/nhịp (1-2-1, 2-2, 3-2, 4-1 ... ) heuristic
  // Đếm nhịp gần nhất: [run1, run2, run3 ...], lấy 3 run cuối
  let m10 = 0.5;
  if (runs.length >= 3) {
    const r = runs.slice(-3);
    const lens = r.map(x => x.len).join('-'); // vd "1-2-1"
    const lastChar = r[r.length - 1].char;
    // Một số “mẫu cầu” thường gặp:
    const biasMap = {
      '1-1': 0.55, '1-2-1': 0.58, '2-1-2': 0.58,
      '2-2': 0.57, '3-1': 0.55, '1-3': 0.55,
      '2-3': 0.56, '3-2': 0.56, '4-1': 0.58, '1-4': 0.58
    };
    // Tạm ánh xạ: nếu lastChar = 'T', nghiêng về T theo bias; ngược lại đảo chiều
    let base = 0.5;
    if (biasMap[lens]) base = biasMap[lens];
    m10 = lastChar === 'T' ? base : (1 - base);
  }

  // Trọng số cho 10 model (tổng = 1)
  const weights = [0.12, 0.12, 0.10, 0.08, 0.12, 0.08, 0.08, 0.10, 0.10, 0.10];
  const models = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10];

  let pT = 0;
  for (let i = 0; i < models.length; i++) pT += models[i] * weights[i];
  pT = clamp01(pT);
  const pX = 1 - pT;

  // Độ tin cậy: dựa trên độ lệch |pT-0.5| + độ dài mẫu
  const conf = clamp01(0.4 + Math.abs(pT - 0.5) * 1.2 + Math.min(n, 20) * 0.01);

  return {
    pT,
    pX,
    confidence: conf,
    details: {
      models: { m1, m2, m3, m4, m5, m6, m7, m8, m9, m10 },
      weights,
      markov: mk,
      streak: st,
      alternation: alt,
      ngram: ng,
      runs,
      balance: balance
    }
  };
}

/**
 * Sinh chuỗi giải thích AI TỔNG HỢP (tiếng Việt)
 */
function buildExplanation(pattern, ens) {
  const txStr = pattern.join('');
  const { streak, markov, alternation, ngram, runs } = ens.details;
  const last3Runs = runs.slice(-3).map(r => `${r.char}:${r.len}`).join(', ');
  const transTT = markov.trans.T.T, transTX = markov.trans.T.X, transXT = markov.trans.X.T, transXX = markov.trans.X.X;

  return [
    `AI TỔNG HỢP phân tích 20 mẫu gần nhất: [${txStr}] (T=Tài, X=Xỉu).`,
    `1) Thống kê: T=${pattern.filter(x=>x==='T').length}, X=${pattern.filter(x=>x==='X').length}, streak cuối: ${streak.char || '-'} x${streak.len}.`,
    `2) Markov(1): từ ${pattern[pattern.length-1] || '-'} → P(T=${markov.probs.T.toFixed(2)}, X=${markov.probs.X.toFixed(2)}), ma trận chuyển: T→T=${transTT}, T→X=${transTX}, X→T=${transXT}, X→X=${transXX}.`,
    `3) Mẫu đảo/luân phiên: điểm alternation ≈ ${alternation.toFixed(2)}.`,
    `4) N-gram (k=3): sau chuỗi đuôi gần nhất có T=${ngram.T}, X=${ngram.X}.`,
    `5) Run gần nhất: ${last3Runs || '-'}.`,
    `6) Ensemble 10 mô hình (không random) cho tỷ lệ: Tài=${(ens.pT*100).toFixed(2)}%, Xỉu=${(ens.pX*100).toFixed(2)}%.`,
    `→ Kết luận AI TỔNG HỢP chọn phương án có xác suất cao hơn, kèm độ tin cậy nội bộ (data-driven).`
  ].join(' ');
}

/**
 * Lấy dữ liệu nguồn, cập nhật pattern, và trả về bản dự đoán cho next_session
 */
async function getPrediction() {
  // 1) Lấy dữ liệu nguồn
  const { data } = await axios.get(SOURCE_URL, { timeout: 12000 });
  // Giả định payload có { list: [ {id, resultTruyenThong, dices, point}, ... ] }
  const list = Array.isArray(data?.list) ? data.list : [];

  if (!list.length) {
    throw new Error('Nguồn không trả về list hợp lệ.');
  }

  // 2) Lấy phiên mới nhất theo id lớn nhất
  const latest = list.reduce((a, b) => (a.id > b.id ? a : b));
  const session = latest.id;
  const dice = latest.dices || [];
  const total = latest.point ?? null;
  const resultRaw = latest.resultTruyenThong || null;
  const resultTX = toTX(resultRaw); // 'T' | 'X' | null
  const result = resultRaw || (resultTX === 'T' ? 'TAI' : resultTX === 'X' ? 'XIU' : null);

  // 3) Cập nhật pattern (lưu tối đa 20)
  const state = loadState();
  if (resultTX) {
    state.pattern.push(resultTX);
    if (state.pattern.length > 20) state.pattern = state.pattern.slice(-20);
    state.lastUpdated = Date.now();
    saveState(state);
  }

  // Tổng gần đây (nếu cần cho 1 số model)
  const recentTotals = list
    .slice(-Math.min(20, list.length))
    .map(x => typeof x.point === 'number' ? x.point : null)
    .filter(x => x !== null);

  // 4) Ensemble 10 model
  const ens = ensembleModels(state.pattern, recentTotals);

  // 5) Quyết định dự đoán cho next_session
  const duDoanTX = ens.pT >= ens.pX ? 'T' : 'X';
  const du_doan = toDisplay(duDoanTX);

  // 6) Độ tin cậy & tỷ lệ
  const do_tin_cay = `${(ens.confidence * 100).toFixed(2)}%`;
  const ty_le = {
    Tai: `${(ens.pT * 100).toFixed(2)}%`,
    Xiu: `${(ens.pX * 100).toFixed(2)}%`
  };

  // 7) Giải thích AI tổng hợp
  const giai_thich = buildExplanation(state.pattern, ens);

  // 8) Kết quả trả về
  return {
    session,
    dice,
    total,
    result, // kết quả phiên hiện tại (nếu có), theo nguồn
    next_session: session + 1,
    du_doan,
    do_tin_cay,
    giai_thich,
    pattern: state.pattern,     // mảng 20 TX gần nhất (ví dụ ["T","X","T",...])
    ty_le,
    id: 'Tele@HoVanThien_Pro'
  };
}

// Endpoint chính
app.get('/api/taixiu/predict', async (req, res) => {
  try {
    const payload = await getPrediction();
    res.json(payload);
  } catch (e) {
    console.error('Lỗi /api/taixiu/predict:', e.message);
    res.status(500).json({ error: e.message || 'Internal Server Error' });
  }
});

// Healthcheck
app.get('/', (req, res) => {
  res.json({ ok: true, name: 'tai-xiu-ai', version: '1.0.0' });
});

app.listen(APP_PORT, () => {
  console.log(`✅ Server chạy tại http://localhost:${APP_PORT}`);
});
