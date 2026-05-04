// ══════════════════════════════════════════════════════════
//  PRESETS, HF_IDS, MODEL_CATS, MODEL_FAMILIES
//  ↑ chargés depuis models.js
// ══════════════════════════════════════════════════════════

// ══════════════════════════════════════════════════════════
const HW = [
  // Apple Silicon — TDP = total chip SoC power
  { name:'Mac mini M4 (16 GB)',         vram:16,  bw:120,  tdp:25,  type:'apple'  },
  { name:'MacBook Pro M4 Pro (24 GB)',  vram:24,  bw:273,  tdp:30,  type:'apple'  },
  { name:'Mac M4 Pro (48 GB)',          vram:48,  bw:273,  tdp:40,  type:'apple'  },
  { name:'Mac M4 Max (64 GB)',          vram:64,  bw:410,  tdp:55,  type:'apple'  },
  { name:'Mac M4 Max (128 GB)',         vram:128, bw:410,  tdp:55,  type:'apple'  },
  { name:'Mac M3 Pro (36 GB)',          vram:36,  bw:150,  tdp:30,  type:'apple'  },
  { name:'Mac M3 Max (96 GB)',          vram:96,  bw:300,  tdp:50,  type:'apple'  },
  { name:'Mac M2 Ultra (192 GB)',       vram:192, bw:800,  tdp:180, type:'apple'  },
  // NVIDIA — TDP = GPU card TDP
  { name:'RTX 4060 Ti (16 GB)',         vram:16,  bw:288,  tdp:165, type:'nvidia' },
  { name:'RTX 4080 (16 GB)',            vram:16,  bw:716,  tdp:320, type:'nvidia' },
  { name:'RTX 4090 (24 GB)',            vram:24,  bw:1008, tdp:450, type:'nvidia' },
  { name:'RTX 5090 (32 GB)',            vram:32,  bw:1792, tdp:575, type:'nvidia' },
  { name:'A100 SXM (40 GB)',            vram:40,  bw:1555, tdp:400, type:'nvidia' },
  { name:'A100 SXM (80 GB)',            vram:80,  bw:2039, tdp:400, type:'nvidia' },
  { name:'H100 SXM (80 GB)',            vram:80,  bw:3350, tdp:700, type:'nvidia' },
  { name:'H200 (141 GB)',               vram:141, bw:4800, tdp:700, type:'nvidia' },
  { name:'B200 (192 GB)',               vram:192, bw:8000, tdp:1000,type:'nvidia' },
  // AMD
  { name:'RX 7900 XTX (24 GB)',         vram:24,  bw:960,  tdp:355, type:'amd'   },
  { name:'AMD MI300X (192 GB)',         vram:192, bw:5300, tdp:750, type:'amd'   },
  // Smartphones — TDP = SoC power, VRAM = RAM (shared)
  { name:'iPhone 15 Pro (8 GB)',        vram:8,   bw:50,   tdp:15,  type:'smartphone' },
  { name:'iPhone 15 Pro Max (8 GB)',   vram:8,   bw:50,   tdp:18,  type:'smartphone' },
  { name:'iPhone 16 Pro (12 GB)',       vram:12,  bw:60,   tdp:18,  type:'smartphone' },
  { name:'Samsung S24 Ultra (12 GB)',   vram:12,  bw:70,   tdp:15,  type:'smartphone' },
  { name:'Google Pixel 8 Pro (12 GB)',  vram:12,  bw:65,   tdp:12,  type:'smartphone' },
  { name:'Samsung S25 Ultra (16 GB)',   vram:16,  bw:80,   tdp:18,  type:'smartphone' },
].sort((a,b)=>a.vram-b.vram);

// Framework overhead: base GB + scale per B params
const FRAMEWORK_OH = {
  llamacpp:  { base:0.3,  scale:0.008, label:'llama.cpp / MLX' },
  ollama:    { base:0.5,  scale:0.010, label:'Ollama'          },
  vllm:      { base:1.5,  scale:0.015, label:'vLLM'            },
  tgi:       { base:1.2,  scale:0.012, label:'HF TGI'          },
  pytorch:   { base:2.0,  scale:0.018, label:'PyTorch/HF'      },
  exllamav2: { base:0.4,  scale:0.009, label:'ExLlamaV2'       },
};

const QMAP = {2:'INT2',3:'INT3',4:'INT4',5:'INT5',6:'INT6',8:'INT8',16:'FP16',32:'FP32'};
const $ = id => document.getElementById(id);

// ─── STATE ───────────────────────────────────────────────
let mode = 'inference';
let hwFilter = 'all';
let dataSource = 'hf';
let totalGB = 0, modelWeightGB = 0;
let llsKey = localStorage.getItem('llsApiKey') || '';
let lastBmData = null;
let trainMult = 5.5;
let optimState = { paged:false, fp8kv:false, int8kv:false, flashattn:false };
let inputMode = 'auto';
let activeCat  = 'all';   // category filter
let activeUsage = 'general'; // usage selector → benchmark key
let compList = [];        // competition items [{key, name, gb, quant}]

// Usage → benchmark score key mapping
const USAGE_BM = {
  general:      'Average',
  math:         'MATH Lvl 5',
  code:         'BBH',
  conversation: 'IFEval',
}; // 'auto' | 'manuel'

// ─── LOCAL STORAGE — CUSTOM MODELS ───────────────────────
function loadCustomModels() {
  try { return JSON.parse(localStorage.getItem('customModels') || '[]') } catch { return [] }
}
function saveCustomModels(arr) {
  localStorage.setItem('customModels', JSON.stringify(arr));
}
function renderCustomModelsSelect() {
  const group = $('customModelsGroup');
  group.innerHTML = '';
  loadCustomModels().forEach(m => {
    const opt = document.createElement('option');
    opt.value  = 'custom_' + m.id;
    opt.textContent = '💾 ' + m.name;
    group.appendChild(opt);
  });
}
function renderCustomModelChips() {
  const list = $('cmList');
  const models = loadCustomModels();
  list.innerHTML = models.length ? '' : '<span style="font-size:.72rem;color:var(--muted)">Aucun modèle sauvegardé.</span>';
  models.forEach(m => {
    const chip = document.createElement('div');
    chip.className = 'cm-chip';
    chip.innerHTML = `<span>${m.name}</span><button class="cm-del" onclick="deleteCustomModel('${m.id}')">✕</button>`;
    list.appendChild(chip);
  });
}
function deleteCustomModel(id) {
  const arr = loadCustomModels().filter(m => m.id !== id);
  saveCustomModels(arr);
  renderCustomModelsSelect();
  renderCustomModelChips();
}
function openAddModal() {
  renderCustomModelChips();
  $('addModal').classList.add('show');
}
function closeAddModal() { $('addModal').classList.remove('show') }
function saveCustomModel() {
  const name = $('am_name').value.trim();
  if (!name) { $('am_error').textContent='❌ Le nom est requis.'; $('am_error').style.display='block'; return }
  const params = parseFloat($('am_params').value);
  if (!params || params <= 0) { $('am_error').textContent='❌ Paramètres invalides.'; $('am_error').style.display='block'; return }
  $('am_error').style.display = 'none';
  const id   = 'u_' + Date.now();
  const hfid = $('am_hfid').value.trim();
  const model = {
    id, name,
    params,
    hidden: parseInt($('am_hidden').value) || 4096,
    layers: parseInt($('am_layers').value) || 32,
    heads:  parseInt($('am_heads').value)  || 32,
    kv:     parseInt($('am_kv').value)     || 8,
    moe:    parseInt($('am_moe').value)    || 100,
    hfid,
  };
  const arr = loadCustomModels();
  arr.push(model);
  saveCustomModels(arr);
  // register in PRESETS + HF_IDS on the fly
  PRESETS['custom_' + id] = { params:model.params, hidden:model.hidden, layers:model.layers, heads:model.heads, kv:model.kv, moe:model.moe };
  if (hfid) HF_IDS['custom_' + id] = hfid;
  renderCustomModelsSelect();
  renderCustomModelChips();
  // reset form
  ['am_name','am_hfid'].forEach(i => $(i).value='');
}

// ─── INPUT MODE (AUTO / MANUEL) ──────────────────────────
function setInputMode(m) {
  inputMode = m;
  $('imbAuto').classList.toggle('active', m==='auto');
  $('imbAuto').classList.toggle('auto',   m==='auto');
  $('imbMan').classList.toggle('active',  m==='manuel');
  $('imbMan').classList.toggle('man',     m==='manuel');
  $('secAuto').style.display   = m==='auto'   ? 'block' : 'none';
  $('secManuel').style.display = m==='manuel' ? 'block' : 'none';
  $('imode-desc').textContent  = m==='auto'
    ? 'Sélectionne un modèle — les paramètres se remplissent automatiquement.'
    : 'Mode expert — saisis tous les paramètres manuellement.';
  // sync sliders on switch
  if (m==='manuel') {
    $('quantBitsM').value = $('quantBits').value;
    $('qBadgeM').textContent = $('qBadge').textContent;
    $('contextLenM').value = $('contextLen').value;
  }
  compute();
}

// ─── READ ACTIVE VALUES (unified for both modes) ─────────
function getInputs() {
  const isAuto = inputMode === 'auto';
  return {
    paramsB:    parseFloat($('paramsB').value)     || 7,
    hidden:     parseInt($('hiddenSize').value)     || 4096,
    layers:     parseInt($('numLayers').value)      || 32,
    heads:      parseInt($('numHeads').value)       || 32,
    kvH:        parseInt($('kvHeads').value)        || 8,
    quant:      parseInt(isAuto ? $('quantBits').value : $('quantBitsM').value) || 4,
    ctx:        parseInt(isAuto ? $('contextLen').value : $('contextLenM').value) || 4096,
    batch:      parseInt($('batchSize').value)      || 1,
    kvBits:     parseInt($('kvBitsIn').value)       || 16,
    moe:        Math.max(1, Math.min(100, parseFloat($('moeActive').value) || 100)),
    fwKey:      isAuto ? ($('frameworkSelAuto').value) : ($('frameworkSel').value),
  };
}

// ─── INIT ────────────────────────────────────────────────
if (llsKey) { $('llsApiKey').value = llsKey; $('keySaved').classList.add('show'); $('llsLockTag').textContent = '✅ Clé active' }

// Load saved custom models into select + PRESETS
loadCustomModels().forEach(m => {
  PRESETS['custom_' + m.id] = { params:m.params, hidden:m.hidden, layers:m.layers, heads:m.heads, kv:m.kv, moe:m.moe };
  if (m.hfid) HF_IDS['custom_' + m.id] = m.hfid;
});
renderCustomModelsSelect();

// Default train method pill
$('trainMethodPills').querySelector('.pill.active')?.classList.add('amber');

$('modelPreset').addEventListener('change', () => {
  const key = $('modelPreset').value;
  const p   = PRESETS[key];
  if (p) {
    $('paramsB').value    = p.params;
    $('hiddenSize').value = p.hidden;
    $('numLayers').value  = p.layers;
    $('numHeads').value   = p.heads;
    $('kvHeads').value    = p.kv;
    $('moeActive').value  = p.moe;
    // Auto-fill context from preset native ctx
    if (p.ctx) {
      $('contextLen').value  = Math.min(p.ctx, 32768); // default to 32K for sanity
      $('contextLenM').value = $('contextLen').value;
    }
    // sync manuel sliders
    $('contextLenM').value = $('contextLen').value;
    $('quantBitsM').value  = $('quantBits').value;
    $('qBadgeM').textContent = $('qBadge').textContent;
  }
  // HF ID for benchmarks
  const hfId = HF_IDS[key] || '';
  $('hfModelId').value = hfId;

  // Cloud-only warning
  const cloudBanner = $('cloudBanner');
  if (cloudBanner) {
    if (p?.cloud) {
      cloudBanner.style.display = 'flex';
      cloudBanner.querySelector('#cloudOllamaCmd').textContent = p.ollama ? `ollama run ${p.ollama}` : 'cloud-only';
    } else {
      cloudBanner.style.display = 'none';
    }
  }

  // Ollama pull command banner
  const ollamaBanner = $('ollamaBanner');
  if (ollamaBanner && p?.ollama && !p?.cloud) {
    ollamaBanner.style.display = 'flex';
    ollamaBanner.querySelector('#ollamaCmd').textContent = `ollama run ${p.ollama}`;
  } else if (ollamaBanner) {
    ollamaBanner.style.display = 'none';
  }

  // Show/hide custom alert only in manuel mode
  $('customAlert').style.display = (key === 'custom' && inputMode === 'manuel') ? 'flex' : 'none';
  showBmState('idle');
  lastBmData = null;
  // MoE alert
  if (p && p.moe < 100) {
    $('moeAlert').style.display = 'flex';
    $('moeAlertTxt').innerHTML = `Modèle <strong>MoE</strong> — tous les experts sont chargés mais seuls <strong>${p.moe}%</strong> sont actifs par token. Le débit est calculé sur les poids actifs uniquement (<strong>Fix v5.2</strong>).`;
  } else { $('moeAlert').style.display = 'none' }
  compute();
});

// Auto-mode inputs
['quantBits','contextLen'].forEach(id => $(id).addEventListener('input', () => {
  const b = $('quantBits').value;
  const label = b + ' bits' + (QMAP[b] ? ' · ' + QMAP[b] : '');
  $('qBadge').textContent  = label;
  $('quantBitsM').value    = b;
  $('qBadgeM').textContent = label;
  $('contextLenM').value   = $('contextLen').value;
  compute();
}));
$('frameworkSelAuto').addEventListener('change', compute);

// Manuel-mode inputs
['quantBitsM','contextLenM'].forEach(id => $(id).addEventListener('input', () => {
  const b = $('quantBitsM').value;
  const label = b + ' bits' + (QMAP[b] ? ' · ' + QMAP[b] : '');
  $('qBadgeM').textContent = label;
  $('quantBits').value     = b;
  $('qBadge').textContent  = label;
  $('contextLen').value    = $('contextLenM').value;
  compute();
}));
['paramsB','hiddenSize','numLayers','numHeads','kvHeads',
 'batchSize','kvBitsIn','moeActive'].forEach(id => $(id).addEventListener('input', compute));
$('frameworkSel').addEventListener('change', compute);

$('customMult').addEventListener('input', () => {
  const v = parseFloat($('customMult').value);
  trainMult = v;
  $('multBadge').textContent = '×' + v.toFixed(1);
  compute();
});

// ─── MODE ────────────────────────────────────────────────
function setMode(m) {
  mode = m;
  $('btnInfer').classList.toggle('active', m==='inference');
  $('btnTrain').classList.toggle('active', m==='training');
  $('trainCard').style.display = m === 'training' ? 'block' : 'none';
  compute();
}

// ─── TRAIN METHOD ────────────────────────────────────────
function setTrainMethod(btn) {
  document.querySelectorAll('#trainMethodPills .pill').forEach(p => p.classList.remove('active','cyan','amber','red','purple'));
  btn.classList.add('active');
  const mult = parseFloat(btn.dataset.mult);
  if (mult === 0) {
    // custom
    btn.classList.add('purple');
    $('customMultRow').style.display = 'block';
    trainMult = parseFloat($('customMult').value);
  } else {
    // pick color by severity
    const cls = mult >= 7 ? 'red' : mult >= 4 ? 'amber' : 'cyan';
    btn.classList.add(cls);
    $('customMultRow').style.display = 'none';
    trainMult = mult;
  }
  compute();
}

// ─── OPTIM TOGGLES ───────────────────────────────────────
function toggleOptim(key) {
  // fp8kv and int8kv are mutually exclusive
  if ((key === 'fp8kv' && optimState.int8kv) || (key === 'int8kv' && optimState.fp8kv)) {
    const other = key === 'fp8kv' ? 'int8kv' : 'fp8kv';
    optimState[other] = false;
    $('opt_'+other).classList.remove('active');
    $('chk_'+other).textContent = '';
  }
  optimState[key] = !optimState[key];
  $('opt_'+key).classList.toggle('active', optimState[key]);
  $('chk_'+key).textContent = optimState[key] ? '✓' : '';
  compute();
}

function getOptimFactor() {
  let kv = 1.0, oh = 1.0;
  if (optimState.paged)   kv *= 0.75;
  if (optimState.fp8kv || optimState.int8kv) kv *= 0.5;
  if (optimState.flashattn) oh *= 0.7;
  return { kv, oh };
}

// ─── COMPUTE ─────────────────────────────────────────────
function compute() {
  const { paramsB, hidden, layers, heads, kvH, quant, ctx, batch, kvBits, moe, fwKey } = getInputs();

  // 1. Weights
  const wGB_raw   = (paramsB * 1e9 * quant / 8) / 1e9;

  // 2. KV cache (GQA-correct) + optim factor
  const headDim   = hidden / heads;
  const { kv: kvFactor, oh: ohFactor } = getOptimFactor();
  const kvGB_base = (2 * layers * kvH * headDim * ctx * batch * (kvBits / 8)) / 1e9;
  const kvGB      = kvGB_base * kvFactor;

  // 3. Dynamic overhead
  const fw    = FRAMEWORK_OH[fwKey] || FRAMEWORK_OH.vllm;
  const ohGB  = (fw.base + fw.scale * paramsB) * ohFactor;
  $('dynOverhead').textContent    = ohGB.toFixed(2) + ' GB';
  $('dynOverheadSub').textContent = fw.label + ' · base ' + fw.base + ' + ' + fw.scale + '×' + paramsB + 'B';

  // 4. Training multiplier on weights
  let wGB = wGB_raw;
  if (mode === 'training') wGB = wGB_raw * trainMult;

  modelWeightGB = wGB_raw; // for throughput
  totalGB = wGB + kvGB + ohGB;

  const fmt = gb => gb >= 10 ? gb.toFixed(1) + ' GB' : gb.toFixed(2) + ' GB';

  $('rW').textContent    = fmt(wGB);
  $('rWSub').textContent = paramsB + 'B params · ' + quant + ' bits' + (mode==='training' ? ' · ×'+trainMult+' '+$('trainMethodPills').querySelector('.pill.active')?.dataset.label:'' );

  const gqaNote = kvH < heads ? ` GQA ${kvH}/${heads}` : ' MHA';
  const optimNote = kvFactor < 1 ? ` · ×${kvFactor.toFixed(2)} optim` : '';
  $('rKV').textContent    = fmt(kvGB);
  $('rKVSub').textContent = `ctx ${ctx.toLocaleString()} · batch ${batch}${gqaNote}${optimNote}`;

  $('rOH').textContent    = fmt(ohGB);
  $('rOHSub').textContent = fw.label;

  $('rTot').textContent    = fmt(totalGB);
  $('rTotSub').textContent = `${totalGB.toFixed(1)} GB · ${(totalGB*1024).toFixed(0)} MiB`;

  const tc = $('rTot');
  tc.style.color = totalGB > 96 ? 'var(--red)' : totalGB > 40 ? 'var(--amber)' : 'var(--cyan)';

  // Throughput viability warning (check if any compatible HW has low BW)
  const worst = HW.filter(h => totalGB <= h.vram).sort((a,b)=>a.bw-b.bw)[0];
  if (worst) {
    const toks = (worst.bw / modelWeightGB * 0.85).toFixed(0);
    if (parseFloat(toks) < 3) {
      $('thrAlert').style.display = 'flex';
      $('thrAlertTxt').innerHTML = `Sur <strong>${worst.name}</strong> (le matériel minimal compatible), le débit estimé est <strong>~${toks} tok/s</strong> — le modèle rentre en mémoire mais <strong>n'est pas utilisable</strong> en pratique.`;
    } else {
      $('thrAlert').style.display = 'none';
    }
  } else {
    $('thrAlert').style.display = 'none';
  }

  renderHW();
  updateEfficiency();
}

// ─── HARDWARE ────────────────────────────────────────────
function setHWFilter(btn, t) {
  hwFilter = t;
  document.querySelectorAll('.fbtn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderHW();
}

function setUsage(btn) {
  activeUsage = btn.dataset.usage;
  document.querySelectorAll('.usage-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderHW();
  updateEfficiency();
}

function renderHW() {
  const list = hwFilter === 'all' ? HW : HW.filter(h => h.type === hwFilter);
  const em = { apple:'🍎', nvidia:'🟢', amd:'🔴' };

  // Compute recommendation scores for compatible HW
  const bmKey = USAGE_BM[activeUsage] || 'Average';
  const bmScore = lastBmData?.scores?.[bmKey] ?? null;

  // Score = 0.5 * norm(tok/s) + 0.3 * norm(bm) + 0.2 * norm(1/tdp)
  // Only for compatible hardware
  const compatible = list.filter(h => totalGB <= h.vram && modelWeightGB > 0);
  let bestIdx = -1;
  if (compatible.length >= 1 && modelWeightGB > 0) {
    const maxToks = Math.max(...compatible.map(h => h.bw / modelWeightGB));
    const minTDP  = Math.min(...compatible.map(h => h.tdp));
    const maxTDP  = Math.max(...compatible.map(h => h.tdp));

    const scored = compatible.map(h => {
      const toks    = h.bw / modelWeightGB * 0.85;
      const normTok = maxToks > 0 ? toks / (maxToks * 0.85) : 0;
      const normBm  = bmScore != null ? bmScore / 100 : 0.5;
      const normEff = maxTDP > minTDP ? (maxTDP - h.tdp) / (maxTDP - minTDP) : 1;
      const score   = 0.50 * normTok + 0.25 * normBm + 0.25 * normEff;
      return { name: h.name, score };
    });
    const best = scored.reduce((a, b) => a.score > b.score ? a : b, scored[0]);
    bestIdx = best ? compatible.findIndex(h => h.name === best.name) : -1;
  }

  let bestCompatIdx = -1; // index in `list`
  if (bestIdx >= 0) {
    bestCompatIdx = list.findIndex(h => h.name === compatible[bestIdx]?.name);
  }

  $('hwGrid').innerHTML = list.map((hw, idx) => {
    const pct    = (totalGB / hw.vram) * 100;
    const st     = totalGB <= hw.vram * 0.82 ? 'fits' : totalGB <= hw.vram ? 'tight' : 'over';
    const lbl    = { fits:'✅ Compatible', tight:'⚠️ Limite', over:'❌ Insuffisant' }[st];
    const barW   = st === 'over' ? 100 : Math.min(pct, 100);
    const isBest = idx === bestCompatIdx && st !== 'over';

    // Throughput
    const toksRaw = modelWeightGB > 0 ? hw.bw / modelWeightGB * 0.85 : 0;
    const toks    = toksRaw >= 1000 ? (toksRaw/1000).toFixed(1)+'k' : toksRaw.toFixed(0);
    const thrCls  = toksRaw >= 15 ? 'fast' : toksRaw >= 4 ? 'medium' : 'slow';
    const thrNote = st === 'over'
      ? '<span class="hw-thr-val" style="color:var(--dim)">N/A</span>'
      : `<span class="hw-thr-val ${thrCls}">~${toks} tok/s</span>`;

    // Energy efficiency
    const tokPerWatt = hw.tdp > 0 && toksRaw > 0 ? (toksRaw / hw.tdp).toFixed(2) : '—';

    return `<div class="hwc ${st}${isBest?' best':''}">
      <div class="hwc-top">
        <span class="hw-name">${em[hw.type]} ${hw.name}</span>
        <span class="hw-badge ${hw.type}">${hw.type.toUpperCase()}</span>
      </div>
      ${isBest ? `<div style="margin-bottom:6px"><span class="rec-badge">🏆 Meilleur choix — ${activeUsage}</span></div>` : ''}
      <div class="hwc-mid">
        <span class="hw-cap">${totalGB.toFixed(1)} / ${hw.vram} GB (${Math.round(Math.min(pct,999))}%)</span>
        <span class="hw-st ${st}">${lbl}</span>
      </div>
      <div class="bar-bg"><div class="bar-fill ${st}" style="width:${barW.toFixed(1)}%"></div></div>
      <div class="hw-thr">
        <span class="hw-thr-lbl">⚡ Débit estimé batch=1</span>
        ${thrNote}
      </div>
      <div class="hw-energy">
        <span class="hw-energy-lbl">🔋 TDP ${hw.tdp}W · Efficience</span>
        <span class="hw-energy-val">${st!=='over'&&toksRaw>0?tokPerWatt+' tok/s/W':'—'}</span>
      </div>
    </div>`;
  }).join('');
}

// ─── CATEGORY FILTER ─────────────────────────────────────
function setCat(btn, cat) {
  activeCat = cat;
  document.querySelectorAll('.cat-pill').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  filterModelSelect();
}

function filterModelSelect() {
  const sel = $('modelPreset');
  const cur = sel.value;
  Array.from(sel.options).forEach(opt => {
    const key  = opt.value;
    const cats = MODEL_CATS[key] || [];
    opt.hidden = activeCat !== 'all' && cats.length > 0 && !cats.includes(activeCat);
  });
  // If current is hidden, reset to first visible
  if (sel.options[sel.selectedIndex]?.hidden) {
    const first = Array.from(sel.options).find(o => !o.hidden && o.value !== '');
    if (first) { sel.value = first.value; sel.dispatchEvent(new Event('change')); }
  }
}

// ─── ANIMATED COUNTER ────────────────────────────────────
function animVal(el) {
  el.classList.remove('num-pop');
  void el.offsetWidth; // reflow
  el.classList.add('num-pop');
}

// ─── COMPETITION SYSTEM ───────────────────────────────────
function populateCompSelect() {
  const sel = $('compModelSel');
  sel.innerHTML = '<option value="">— Choisir un modèle —</option>';
  Object.keys(PRESETS).forEach(key => {
    if (key === 'custom') return;
    const opt = document.createElement('option');
    opt.value = key;
    // Find label from modelPreset select
    const src = $('modelPreset').querySelector(`option[value="${key}"]`);
    opt.textContent = src ? src.textContent.replace(/^[💾🔧]\s*/,'') : key;
    sel.appendChild(opt);
  });
}

function addToComp() {
  const key   = $('compModelSel').value;
  const quant = parseInt($('compQuantSel').value) || 4;
  if (!key || !PRESETS[key]) return;
  if (compList.length >= 8) { alert('Maximum 8 modèles dans le comparateur.'); return }
  if (compList.find(c => c.key === key && c.quant === quant)) return; // duplicate

  const p       = PRESETS[key];
  const wGB     = (p.params * 1e9 * quant / 8) / 1e9;
  const headDim = p.hidden / p.heads;
  const kvGB    = (2 * p.layers * p.kv * headDim * 4096 * 1 * (16/8)) / 1e9;
  const gb      = parseFloat((wGB + kvGB + 0.5).toFixed(1));

  const src  = $('modelPreset').querySelector(`option[value="${key}"]`);
  const name = (src ? src.textContent.replace(/^[💾🔧]\s*/,'') : key) + ` (${quant}b)`;

  compList.push({ key, name, gb, quant, params: p.params });
  renderComp();
}

function removeFromComp(idx) {
  compList.splice(idx, 1);
  renderComp();
}

function clearComp() {
  compList = [];
  renderComp();
}

const COMP_COLORS = ['#22d3ee','#10b981','#f59e0b','#a78bfa','#ef4444','#fb923c','#34d399','#818cf8'];

function renderComp() {
  // Chips
  $('compChips').innerHTML = compList.map((c, i) =>
    `<div class="comp-chip" style="border-color:${COMP_COLORS[i%COMP_COLORS.length]}33">
      <span class="comp-legend-dot" style="background:${COMP_COLORS[i%COMP_COLORS.length]}"></span>
      <span class="comp-chip-name">${c.name}</span>
      <span class="comp-chip-mem">${c.gb} GB</span>
      <button class="comp-chip-del" onclick="removeFromComp(${i})">✕</button>
    </div>`
  ).join('');

  if (compList.length < 2) {
    $('compEmpty').style.display = 'flex';
    $('compChart').style.display = 'none';
    return;
  }
  $('compEmpty').style.display = 'none';
  $('compChart').style.display = 'block';
  renderCompChart();
}

function renderCompChart() {
  const W = Math.max(400, ($('compChart').offsetWidth || 600));
  const H = 240;
  const PAD = { top:20, right:20, bottom:50, left:58 };
  const chartW = W - PAD.left - PAD.right;
  const chartH = H - PAD.top  - PAD.bottom;

  const maxGB    = Math.max(...compList.map(c => c.gb), 1);
  const barWidth = Math.floor(chartW / compList.length * 0.62);
  const gap      = chartW / compList.length;

  // Gridlines
  const gridCount = 5;
  let svgContent = '';
  for (let i = 0; i <= gridCount; i++) {
    const y   = PAD.top + chartH - (i / gridCount) * chartH;
    const val = ((i / gridCount) * maxGB).toFixed(0);
    svgContent += `<line x1="${PAD.left}" x2="${PAD.left+chartW}" y1="${y}" y2="${y}" stroke="var(--b1)" stroke-width="1"/>
    <text x="${PAD.left-6}" y="${y+4}" class="comp-axis-lbl" text-anchor="end">${val}</text>`;
  }

  // Bars
  compList.forEach((c, i) => {
    const x   = PAD.left + gap * i + gap/2 - barWidth/2;
    const pct = c.gb / maxGB;
    const bH  = Math.max(3, pct * chartH);
    const y   = PAD.top + chartH - bH;
    const col = COMP_COLORS[i % COMP_COLORS.length];

    svgContent += `
    <rect x="${x}" y="${y}" width="${barWidth}" height="${bH}" fill="${col}" opacity="0.85" rx="4">
      <animate attributeName="height" from="0" to="${bH}" dur="0.5s" begin="${i*0.08}s" fill="freeze"/>
      <animate attributeName="y" from="${PAD.top+chartH}" to="${y}" dur="0.5s" begin="${i*0.08}s" fill="freeze"/>
    </rect>
    <text x="${x + barWidth/2}" y="${y - 5}" class="comp-bar-lbl" text-anchor="middle" fill="${col}">${c.gb}</text>
    <text x="${x + barWidth/2}" y="${H - PAD.bottom + 16}" class="comp-axis-lbl" text-anchor="middle" fill="var(--muted)">${c.name.split(' ')[0]}</text>
    <text x="${x + barWidth/2}" y="${H - PAD.bottom + 29}" class="comp-axis-lbl" text-anchor="middle" fill="var(--dim)">${c.quant}b</text>`;
  });

  // Axis label
  svgContent += `<text x="${PAD.left - 42}" y="${PAD.top + chartH/2}" class="comp-axis-lbl" text-anchor="middle" fill="var(--muted)" transform="rotate(-90,${PAD.left-42},${PAD.top+chartH/2})">RAM / VRAM (GB)</text>`;

  $('compSvg').setAttribute('viewBox', `0 0 ${W} ${H}`);
  $('compSvg').setAttribute('width', '100%');
  $('compSvg').innerHTML = svgContent;

  // Legend
  $('compLegend').innerHTML = compList.map((c,i) =>
    `<div class="comp-legend-item">
      <span class="comp-legend-dot" style="background:${COMP_COLORS[i%COMP_COLORS.length]}"></span>
      <span>${c.name} — ${c.gb} GB</span>
    </div>`
  ).join('');
}

// ─── SOURCE TOGGLE ───────────────────────────────────────
function setSource(src) {
  dataSource = src;
  $('srcHF').classList.toggle('active',  src==='hf');
  $('srcLLS').classList.toggle('active', src==='lls');
  $('keyPanel').classList.toggle('show', src==='lls');
  showBmState('idle');
  lastBmData = null;
}

function saveLLSKey() {
  const k = $('llsApiKey').value.trim();
  if (!k) return;
  llsKey = k;
  localStorage.setItem('llsApiKey', k);
  $('keySaved').classList.add('show');
  $('llsLockTag').textContent = '✅ Clé active';
}

// ─── BENCHMARK STATES ────────────────────────────────────
function showBmState(state, msg) {
  ['bmIdle','bmLoading','bmEmpty','bmError','bmHubFallback'].forEach(id => $(id).style.display='none');
  $('bmFound').classList.remove('show');
  if (state==='idle')        $('bmIdle').style.display='flex';
  if (state==='loading')     $('bmLoading').style.display='flex';
  if (state==='empty')       $('bmEmpty').style.display='flex';
  if (state==='error')       { $('bmError').style.display='flex'; if(msg) $('bmErrorMsg').textContent='⚠️ '+msg }
  if (state==='hubfallback') $('bmHubFallback').style.display='block';
  if (state==='found')       $('bmFound').classList.add('show');
}

async function fetchBenchmarks() {
  const modelId = $('hfModelId').value.trim();
  if (!modelId) { showBmState('error','Saisis un ID HuggingFace.'); return }
  if (dataSource==='lls') { if (!llsKey) { showBmState('error','Clé API llm-stats manquante.'); return } return fetchLLSBenchmarks(modelId) }
  return fetchHFBenchmarks(modelId);
}

// ─── HELPERS ─────────────────────────────────────────────
// Strip quantization & variant suffixes to get a clean base name for leaderboard search
function cleanForLeaderboard(name) {
  return name
    // quantization formats
    .replace(/[-_]?(fp8|fp16|fp32|bf16|int4|int8|w4a16|w8a16|awq|gptq|gguf|ggml|exl2|hqq|nf4|q4|q8)/gi, '')
    // variant suffixes
    .replace(/[-_]?(instruct|chat|it|hf|turbo|plus|ultra|lite|mini|nano|base|v\d[\d.]*)/gi, '')
    // trailing separators
    .replace(/[-_\.]+$/g, '')
    .trim();
}

// Safe fetch with timeout — avoids hanging on HF API flakiness
async function safeFetch(url, opts = {}, timeoutMs = 8000) {
  const ctrl = new AbortController();
  const tid  = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...opts, signal: ctrl.signal });
    clearTimeout(tid);
    return res;
  } catch(e) {
    clearTimeout(tid);
    throw e;
  }
}

// ─── MAIN FETCH (4-strategy cascade) ─────────────────────
async function fetchHFBenchmarks(modelId) {
  showBmState('loading');
  $('fetchBtn').disabled = true;
  $('fetchIcon').textContent = '⏳';

  const rawName   = modelId.split('/').pop();          // e.g. "Qwen3.6-35B-A3B-FP8"
  const cleanName = cleanForLeaderboard(rawName);      // e.g. "Qwen3.6-35B-A3B"
  const baseName  = cleanName.split('-')[0];           // e.g. "Qwen3.6"
  const HF_SEARCH = 'https://datasets-server.huggingface.co/search'
                  + '?dataset=open-llm-leaderboard%2Fcontents&config=default&split=train'
                  + '&offset=0&length=10&query=';

  const hdrs = { Accept: 'application/json' };

  try {
    // ── STRATEGY 1 : search with raw model name ──────────
    let rows = await trySearch(HF_SEARCH + encodeURIComponent(rawName), hdrs);

    // ── STRATEGY 2 : search with cleaned name (no quant suffix) ──
    if (!rows && cleanName !== rawName)
      rows = await trySearch(HF_SEARCH + encodeURIComponent(cleanName), hdrs);

    // ── STRATEGY 3 : search with base name only ───────────
    if (!rows && baseName.length > 3)
      rows = await trySearch(HF_SEARCH + encodeURIComponent(baseName), hdrs);

    // ── STRATEGY 4 : HF Hub API fallback (model metadata) ─
    if (!rows) {
      await showHubFallback(modelId);
      return;
    }

    processHFRows(rows, modelId);

  } catch(e) {
    // Network completely down
    showBmState('error', 'Impossible de contacter HuggingFace — vérifie ta connexion. (' + e.message + ')');
  } finally {
    $('fetchBtn').disabled  = false;
    $('fetchIcon').textContent = '🔍';
  }
}

// Returns rows[] or null (never throws)
async function trySearch(url, hdrs) {
  try {
    const res = await safeFetch(url, { headers: hdrs });
    if (!res.ok) return null;           // 4xx, 5xx → try next strategy
    const data = await res.json();
    return data.rows?.length ? data.rows : null;
  } catch { return null }
}

// Strategy 4 : HF Hub model card API
async function showHubFallback(modelId) {
  try {
    const res = await safeFetch(
      `https://huggingface.co/api/models/${encodeURIComponent(modelId)}`,
      { headers: { Accept: 'application/json' } }
    );

    if (!res.ok) {
      // Model doesn't exist on HF either
      showBmState('empty');
      return;
    }

    const m = await res.json();

    // Populate Hub fallback UI
    $('bmHubLabel').textContent = m.id || modelId;
    $('bmSubmitLink').href = `https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard`;

    const fmt_num = n => n >= 1000 ? (n/1000).toFixed(1)+'k' : String(n ?? '—');
    const tags    = (m.tags || []).filter(t => !t.includes(':') && t.length < 30).slice(0, 6);
    const metaItems = [
      { icon:'👤', lbl:'Auteur',       val: m.author || m.id?.split('/')[0] || '—'  },
      { icon:'📥', lbl:'Téléchargements/mois', val: fmt_num(m.downloads)            },
      { icon:'❤️', lbl:'Likes',         val: fmt_num(m.likes)                       },
      { icon:'📅', lbl:'Mis à jour',    val: m.lastModified?.slice(0,10) || '—'     },
      { icon:'🔑', lbl:'Pipeline',      val: m.pipeline_tag || '—'                   },
      { icon:'📦', lbl:'Taille',        val: m.safetensors?.total
                                              ? (m.safetensors.total/1e9).toFixed(1)+'B params'
                                              : '—'                                   },
    ];

    $('bmHubMeta').innerHTML = metaItems.map(i =>
      `<div class="sc">
        <div class="sc-name">${i.icon} ${i.lbl}</div>
        <div style="font-size:.85rem;color:var(--txt)">${i.val}</div>
      </div>`
    ).join('') + (tags.length ? `<div class="sc" style="grid-column:1/-1;text-align:left">
      <div class="sc-name">🏷️ Tags</div>
      <div style="display:flex;flex-wrap:wrap;gap:5px;margin-top:6px">${
        tags.map(t=>`<span class="tag" style="font-size:.68rem">${t}</span>`).join('')
      }</div>
    </div>` : '');

    showBmState('hubfallback');

  } catch {
    // Even Hub API failed — show empty with helpful context
    showBmState('empty');
  }
}

function processHFRows(rows, requestedId) {
  const target = requestedId.toLowerCase();
  let best = rows[0];
  for (const r of rows) {
    const fn = (r.row.fullname||r.row.model||r.row.id||'').toLowerCase();
    if (fn===target||fn.includes(target.split('/').pop())) { best=r; break }
  }
  const row = best.row;
  const scores = {
    'Average':    parseFloat(row['Average ⬆️']??row['average']??row['Average']??null),
    'IFEval':     parseFloat(row['IFEval']??null),
    'BBH':        parseFloat(row['BBH']??null),
    'MATH Lvl 5': parseFloat(row['MATH Lvl 5']??row['MATH']??null),
    'GPQA':       parseFloat(row['GPQA']??null),
    'MuSR':       parseFloat(row['MuSR']??null),
    'MMLU-PRO':   parseFloat(row['MMLU-PRO']??row['MMLU']??null),
  };
  const modelLabel = row.fullname||row.model||row.id||requestedId;
  lastBmData = { scores, modelLabel };
  renderBenchmarks(scores, modelLabel);
}

async function fetchLLSBenchmarks(modelId) {
  showBmState('loading');
  $('fetchBtn').disabled=true; $('fetchIcon').textContent='⏳';
  try {
    const BASE = 'https://api.llm-stats.com/v1';
    const H = { 'Authorization':'Bearer '+llsKey, 'Content-Type':'application/json' };
    const [metaRes, scoresRes] = await Promise.all([
      fetch(`${BASE}/models/${encodeURIComponent(modelId)}`, {headers:H}),
      fetch(`${BASE}/leaderboard/all?model=${encodeURIComponent(modelId)}`, {headers:H})
    ]);
    if (metaRes.status===401) { showBmState('error','Clé API invalide.'); return }
    if (!metaRes.ok) throw new Error('/models: HTTP '+metaRes.status);
    const meta=await metaRes.json(), s=await scoresRes.json();
    const mapped = {
      'Average':    s.average??null, 'IFEval':s.ifeval??null, 'BBH':s.bbh??null,
      'MATH Lvl 5': s.math??null,    'GPQA':s.gpqa??null,     'MuSR':s.musr??null,
      'MMLU-PRO':   s.mmlu_pro??s.mmlu??null,
    };
    lastBmData = { scores:mapped, modelLabel: meta.name||meta.id||modelId };
    renderBenchmarks(mapped, lastBmData.modelLabel);
  } catch(e) { showBmState('error','llm-stats API : '+e.message) }
  finally { $('fetchBtn').disabled=false; $('fetchIcon').textContent='🔍' }
}

function renderBenchmarks(scores, modelLabel) {
  $('bmModelLabel').textContent = modelLabel;
  $('bmScores').innerHTML = Object.entries(scores).map(([name,val]) => {
    const isAvg = name==='Average';
    const has   = val!==null&&!isNaN(val);
    const pct   = has?Math.min(val,100):0;
    const col   = has?(pct>=65?'var(--green)':pct>=45?'var(--amber)':'var(--red)'):'var(--muted)';
    return `<div class="sc ${isAvg?'avg':''}">
      <div class="sc-name">${name}</div>
      <div class="sc-val" style="${isAvg?'':'color:'+col}">${has?val.toFixed(1)+'%':'—'}</div>
      <div class="sc-bar"><div class="sc-fill" style="width:${pct}%;background:${col}"></div></div>
    </div>`;
  }).join('');
  showBmState('found');
  updateEfficiency();
}

function updateEfficiency() {
  if (!lastBmData||totalGB<=0) return;
  const mp = lastBmData.scores['MMLU-PRO'];
  if (mp===null||isNaN(mp)) { $('effVal').textContent='—'; $('effDesc').textContent='MMLU-PRO non disponible'; return }
  $('effVal').textContent = (mp/totalGB).toFixed(2);
  $('effDesc').textContent = `MMLU-PRO ${mp.toFixed(1)}% ÷ ${totalGB.toFixed(1)} GB RAM/VRAM`;
}

// ─── BOOT ─────────────────────────────────────────────────

// 1. Build <select> from MODEL_FAMILIES (organized by family)
function buildModelSelect() {
  const sel = $('modelPreset');
  // insert family optgroups before the customModelsGroup
  const anchor = $('customModelsGroup');
  MODEL_FAMILIES.forEach(fam => {
    const grp = document.createElement('optgroup');
    grp.label = fam.emoji + ' ' + fam.label;
    fam.keys.forEach(key => {
      const p = PRESETS[key];
      if (!p) return;
      const opt = document.createElement('option');
      opt.value = key;
      const cloudTag = p.cloud ? ' ☁️' : '';
      const moeTag   = p.moe < 100 ? ` MoE${p.moe}%` : '';
      const sizeTag  = p.params >= 1 ? ` ${p.params}B` : ` ${Math.round(p.params*1000)}M`;
      opt.textContent = `${sizeTag} — ${p.note || ''}${moeTag}${cloudTag}`.trim();
      // Better label: use family name + size
      opt.textContent = key.replace(/_/g,' ').replace(/(\d)/,' $1').trim() + (p.cloud?' ☁️':'') + (p.moe<100?' 🔀':'');
      // Clean label built from family
      const famShort = fam.label.split(' — ')[0];
      opt.textContent = `${famShort} ${p.params >= 1 ? p.params+'B' : Math.round(p.params*1000)+'M'}${p.moe<100?' [MoE]':''}${p.cloud?' ☁️':''}`;
      if (p.note) opt.title = p.note;
      grp.appendChild(opt);
    });
    sel.insertBefore(grp, anchor);
  });
  // default selection: llama3_8b
  sel.value = 'llama3_8b';
}
buildModelSelect();

// 2. Set PEFT as default active train method
document.querySelectorAll('#trainMethodPills .pill').forEach(p => p.classList.remove('active','cyan','amber','red','purple'));
const peft = document.querySelector('#trainMethodPills [data-label="PEFT"]');
if(peft){ peft.classList.add('active','amber'); trainMult=5.5; }

// 3. Init input mode to Auto
setInputMode('auto');

// 4. Trigger preset load
$('modelPreset').dispatchEvent(new Event('change'));

// 5. Close modal on overlay click
$('addModal').addEventListener('click', e => { if(e.target===$('addModal')) closeAddModal(); });

// 6. Populate competition select
populateCompSelect();

// 7. Animated counters on result values
['rW','rKV','rOH','rTot'].forEach(id => {
  new MutationObserver(() => animVal($(id))).observe($(id), { childList:true, characterData:true, subtree:true });
});
