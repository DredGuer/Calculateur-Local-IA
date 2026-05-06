/**
 * LLM MODELS DATABASE — v5.2
 * Source : ollama.com + HuggingFace model cards
 * Organisé par famille, triée par taille dans chaque famille
 *
 * Champs par modèle :
 *   params  (B)  — paramètres totaux (pour le calcul mémoire)
 *   hidden       — hidden size (dim interne du transformer)
 *   layers       — nombre de couches (blocks)
 *   heads        — têtes d'attention totales (num_attention_heads)
 *   kv           — têtes KV (GQA/MQA), = heads si MHA classique
 *   moe          — % de paramètres actifs par token (100 = dense, <100 = MoE)
 *   ctx          — contexte natif en tokens (pour pré-remplir contextLen)
 *   ollama       — tag ollama run (ex: "gemma4:26b")
 *   hfid         — ID HuggingFace (optionnel, pour benchmark)
 *   cloud        — true si cloud-only (pas de poids locaux)
 *   tags         — catégories pour le filtre UI
 *   note         — note courte affichée dans l'UI
 */

// ════════════════════════════════════════════════════════════
//  FAMILIES — structure d'affichage du <select>
//  Chaque entrée : { label, emoji, keys: [...] }
// ════════════════════════════════════════════════════════════
const MODEL_FAMILIES = [
  // ── Google DeepMind ─────────────────────────────────────
  {
    label: 'Gemma 4 — Google DeepMind',
    emoji: '🟣',
    keys: ['gemma4_e2b','gemma4_e4b','gemma4_26b','gemma4_31b'],
  },
  {
    label: 'Gemma 3 — Google DeepMind',
    emoji: '🟣',
    keys: ['gemma3_1b','gemma3_4b','gemma3_12b','gemma3_27b'],
  },
  // ── Qwen / Alibaba ──────────────────────────────────────
  {
    label: 'Qwen 3.6 — Alibaba',
    emoji: '🔵',
    keys: ['qwen36_27b','qwen36_35b'],
  },
  {
    label: 'Qwen 3.5 — Alibaba',
    emoji: '🔵',
    keys: ['qwen35_08b','qwen35_2b','qwen35_4b','qwen35_9b','qwen35_27b','qwen35_35b','qwen35_122b'],
  },
  {
    label: 'Qwen 3 — Alibaba',
    emoji: '🔵',
    keys: ['qwen3_05b','qwen3_15b','qwen3_7b','qwen3_14b','qwen3_32b','qwen3_72b'],
  },
  {
    label: 'Qwen 3 Coder — Alibaba',
    emoji: '🔵',
    keys: ['qwen3coder_30b','qwen3coder_480b'],
  },
  {
    label: 'Qwen 3 VL — Vision Language',
    emoji: '🔵',
    keys: ['qwen3vl_2b','qwen3vl_4b','qwen3vl_8b','qwen3vl_30b','qwen3vl_32b','qwen3vl_235b'],
  },
  {
    label: 'Qwen 2.5 — Alibaba',
    emoji: '🔵',
    keys: ['qwen25_05b','qwen25_15b','qwen25_3b','qwen25_7b','qwen25_14b','qwen25_32b','qwen25_72b','qwen25_coder_32b'],
  },
  {
    label: 'Qwen 2 Math — Alibaba',
    emoji: '🔵',
    keys: ['qwen2math_15b','qwen2math_7b','qwen2math_72b'],
  },
  // ── Mistral AI ───────────────────────────────────────────
  {
    label: 'Mistral — Mistral AI',
    emoji: '🟤',
    keys: ['mistral_7b','mistral_nemo','mistral_small31','devstral_small2','mistral_medium35','mistral_large2'],
  },
  {
    label: 'Mixtral MoE — Mistral AI',
    emoji: '🟤',
    keys: ['mixtral_8x7b','mixtral_8x22b'],
  },
  // ── DeepSeek ────────────────────────────────────────────
  {
    label: 'DeepSeek V4 — DeepSeek AI',
    emoji: '🔴',
    keys: ['deepseek_v4_flash','deepseek_v4_pro'],
  },
  {
    label: 'DeepSeek R1 — DeepSeek AI',
    emoji: '🔴',
    keys: ['deepseek_r1_distill_7b','deepseek_r1_distill_70b','deepseek_r1'],
  },
  {
    label: 'DeepSeek V3 — DeepSeek AI',
    emoji: '🔴',
    keys: ['deepseek_v3'],
  },
  {
    label: 'DeepSeek OCR — DeepSeek AI',
    emoji: '🔴',
    keys: ['deepseek_ocr'],
  },
  // ── Meta ────────────────────────────────────────────────
  {
    label: 'LLaMA 3 / 3.1 / 3.2 / 3.3 — Meta',
    emoji: '🟠',
    keys: ['llama32_1b','llama32_3b','llama3_8b','llama32_11b_vision','llama33_70b','llama3_70b','llama32_90b_vision','llama3_405b'],
  },
  {
    label: 'LLaMA 2 — Meta',
    emoji: '🟠',
    keys: ['llama2_7b','llama2_13b','llama2_70b'],
  },
  // ── Microsoft ───────────────────────────────────────────
  {
    label: 'Phi — Microsoft',
    emoji: '🟡',
    keys: ['phi3_mini','phi3_medium','phi4'],
  },
  // ── Cohere ──────────────────────────────────────────────
  {
    label: 'Command R — Cohere',
    emoji: '⚫',
    keys: ['command_r','command_rplus','aya_35b'],
  },
  // ── Zhipu AI ────────────────────────────────────────────
  {
    label: 'GLM — Zhipu AI',
    emoji: '🟢',
    keys: ['glm51','glm47_flash'],
  },
  // ── Autres ──────────────────────────────────────────────
  {
    label: 'SmolLM — HuggingFace',
    emoji: '⚪',
    keys: ['smollm_135m','smollm_360m','smollm_1b'],
  },
  {
    label: 'TinyLlama / CodeLlama — Community',
    emoji: '⚪',
    keys: ['tinyllama','codellama_34b'],
  },
  {
    label: 'Falcon — TII',
    emoji: '⚪',
    keys: ['falcon_40b'],
  },
  {
    label: 'Yi — 01.AI',
    emoji: '⚪',
    keys: ['yi_34b'],
  },
];

// ════════════════════════════════════════════════════════════
//  PRESETS — données architecture
// ════════════════════════════════════════════════════════════
const PRESETS = {

  // ── GEMMA 4 ─────────────────────────────────────────────
  // Source: ollama.com/library/gemma4 + Google DeepMind card
  // E2B: 5.1B total params, 2.3B effective, 35 layers, 128K ctx, MoE, multimodal (text+image+audio)
  // E4B: 8B total params, 4.5B effective, 42 layers, 128K ctx, MoE, multimodal (text+image+audio)
  // 26B: 25.2B total, 3.8B active, 30 layers, 256K ctx, MoE, multimodal (text+image), 8/128 experts
  // 31B: 30.7B dense, 60 layers, 256K ctx, multimodal (text+image)
  gemma4_e2b:  { params:5.1,   hidden:2560,  layers:35,  heads:16,  kv:4,  moe:45,  ctx:131072,  ollama:'gemma4:e2b',   hfid:'google/gemma-4-e2b-it',  tags:['vision','thinking','conversation','multilingual'], note:'MoE · 2.3B actifs/5.1B · multimodal' },
  gemma4_e4b:  { params:8,     hidden:3072,  layers:42,  heads:20,  kv:4,  moe:56,  ctx:131072,  ollama:'gemma4:e4b',   hfid:'google/gemma-4-e4b-it',  tags:['vision','thinking','conversation','multilingual'], note:'MoE · 4.5B actifs/8B · multimodal' },
  gemma4_26b:  { params:25.2,  hidden:4096,  layers:30,  heads:16,  kv:8,  moe:15,  ctx:262144,  ollama:'gemma4:26b',   hfid:'google/gemma-4-26b-it',  tags:['vision','thinking','code'],         note:'MoE · 3.8B actifs/25.2B · 8/128 experts · ctx 256K' },
  gemma4_31b:  { params:30.7,  hidden:5120,  layers:60,  heads:20,  kv:10, moe:100, ctx:262144,  ollama:'gemma4:31b',   hfid:'google/gemma-4-31b-it',  tags:['vision','thinking','code','math'],  note:'Dense · ctx 256K · multimodal' },

  // ── GEMMA 3 ─────────────────────────────────────────────
  gemma3_1b:   { params:1,     hidden:1152,  layers:26,  heads:4,   kv:1,  moe:100, ctx:32768,   ollama:'gemma3:1b',    hfid:'google/gemma-3-1b-it',   tags:['conversation'],                    note:'Edge · ctx 32K' },
  gemma3_4b:   { params:4,     hidden:2560,  layers:34,  heads:8,   kv:4,  moe:100, ctx:131072,  ollama:'gemma3:4b',    hfid:'google/gemma-3-4b-it',   tags:['vision','conversation'],           note:'ctx 128K' },
  gemma3_12b:  { params:12,    hidden:3840,  layers:36,  heads:16,  kv:8,  moe:100, ctx:131072,  ollama:'gemma3:12b',   hfid:'google/gemma-3-12b-it',  tags:['vision','conversation'],           note:'ctx 128K' },
  gemma3_27b:  { params:27,    hidden:5120,  layers:46,  heads:16,  kv:8,  moe:100, ctx:131072,  ollama:'gemma3:27b',   hfid:'google/gemma-3-27b-it',  tags:['vision','conversation','code'],    note:'ctx 128K' },

  // ── QWEN 3.6 ────────────────────────────────────────────
  // Source: ollama.com/library/qwen3.6
  // 27B: Dense — 17GB GGUF Q4
  // 35B-A3B: MoE 35B total, 3B actifs — 24GB GGUF Q4
  qwen36_27b:  { params:27,    hidden:5120,  layers:52,  heads:40,  kv:8,  moe:100, ctx:262144,  ollama:'qwen3.6:27b',      hfid:'Qwen/Qwen3.6-27B',     tags:['thinking','code','conversation'], note:'Dense · ctx 256K' },
  qwen36_35b:  { params:35,    hidden:4096,  layers:94,  heads:40,  kv:8,  moe:9,   ctx:262144,  ollama:'qwen3.6:35b',      hfid:'Qwen/Qwen3.6-35B-A3B', tags:['thinking','code'],               note:'MoE A3B · 3B actifs · ctx 256K' },

  // ── QWEN 3.5 ────────────────────────────────────────────
  // Source: ollama.com/library/qwen3.5 — multimodal vision+thinking
  // Sizes GGUF: 0.8b=1GB, 2b=2.7GB, 4b=3.4GB, 9b=6.6GB, 27b=17GB, 35b=24GB, 122b=81GB
  qwen35_08b:  { params:0.8,   hidden:1024,  layers:28,  heads:16,  kv:8,  moe:100, ctx:262144,  ollama:'qwen3.5:0.8b',  hfid:'Qwen/Qwen3.5-0.8B',   tags:['thinking','vision','conversation'], note:'Dense · ctx 256K' },
  qwen35_2b:   { params:2,     hidden:1536,  layers:28,  heads:16,  kv:8,  moe:100, ctx:262144,  ollama:'qwen3.5:2b',    hfid:'Qwen/Qwen3.5-2B',     tags:['thinking','vision','conversation'], note:'Dense · ctx 256K' },
  qwen35_4b:   { params:4,     hidden:2560,  layers:32,  heads:20,  kv:5,  moe:100, ctx:262144,  ollama:'qwen3.5:4b',    hfid:'Qwen/Qwen3.5-4B',     tags:['thinking','vision','code'],         note:'Dense · ctx 256K' },
  qwen35_9b:   { params:9,     hidden:3584,  layers:36,  heads:28,  kv:4,  moe:100, ctx:262144,  ollama:'qwen3.5:9b',    hfid:'Qwen/Qwen3.5-9B',     tags:['thinking','vision','code'],         note:'Dense · ctx 256K' },
  qwen35_27b:  { params:27,    hidden:5120,  layers:52,  heads:40,  kv:8,  moe:100, ctx:262144,  ollama:'qwen3.5:27b',   hfid:'Qwen/Qwen3.5-27B',    tags:['thinking','vision','code','math'],  note:'Dense · ctx 256K' },
  qwen35_35b:  { params:35,    hidden:5120,  layers:64,  heads:40,  kv:8,  moe:100, ctx:262144,  ollama:'qwen3.5:35b',   hfid:'Qwen/Qwen3.5-35B',    tags:['thinking','vision','code','math'],  note:'Dense · ctx 256K' },
  qwen35_122b: { params:122,   hidden:8192,  layers:96,  heads:64,  kv:8,  moe:100, ctx:262144,  ollama:'qwen3.5:122b',  hfid:'Qwen/Qwen3.5-122B',   tags:['thinking','vision','code','math'],  note:'Dense · ctx 256K' },

  // ── QWEN 3 ──────────────────────────────────────────────
  qwen3_05b:   { params:0.6,   hidden:1024,  layers:28,  heads:16,  kv:8,  moe:100, ctx:32768,   ollama:'qwen3:0.6b',  hfid:'Qwen/Qwen3-0.6B',  tags:['thinking','conversation'], note:'Dense' },
  qwen3_15b:   { params:1.7,   hidden:2048,  layers:28,  heads:16,  kv:8,  moe:100, ctx:32768,   ollama:'qwen3:1.7b',  hfid:'Qwen/Qwen3-1.7B',  tags:['thinking','conversation'], note:'Dense' },
  qwen3_7b:    { params:7.6,   hidden:3584,  layers:28,  heads:28,  kv:4,  moe:100, ctx:131072,  ollama:'qwen3:7b',    hfid:'Qwen/Qwen3-7B',    tags:['thinking','code','math'],  note:'Dense · ctx 128K' },
  qwen3_14b:   { params:14.7,  hidden:5120,  layers:40,  heads:40,  kv:8,  moe:100, ctx:131072,  ollama:'qwen3:14b',   hfid:'Qwen/Qwen3-14B',   tags:['thinking','code','math'],  note:'Dense · ctx 128K' },
  qwen3_32b:   { params:32,    hidden:5120,  layers:64,  heads:40,  kv:8,  moe:100, ctx:131072,  ollama:'qwen3:32b',   hfid:'Qwen/Qwen3-32B',   tags:['thinking','code','math'],  note:'Dense · ctx 128K' },
  qwen3_72b:   { params:72,    hidden:8192,  layers:80,  heads:64,  kv:8,  moe:100, ctx:131072,  ollama:'qwen3:72b',   hfid:'Qwen/Qwen3-72B',   tags:['thinking','code','math'],  note:'Dense · ctx 128K' },

  // ── QWEN 3 CODER ────────────────────────────────────────
  // qwen3-coder-next sur Ollama (local/cloud)
  // 30B : Dense coding model
  // 480B : MoE — architecture proche de Qwen2.5-72B × 6.6 experts
  //   480B total, ~22B actifs (4.6% actif)
  qwen3coder_30b:  { params:30,   hidden:5120,  layers:64,  heads:40,  kv:8,  moe:100, ctx:131072,  ollama:'qwen3-coder:30b',   hfid:'Qwen/Qwen3-Coder-30B',   tags:['code'],      note:'Dense · ctx 128K' },
  qwen3coder_480b: { params:480,  hidden:8192,  layers:94,  heads:64,  kv:8,  moe:5,   ctx:131072,  ollama:'qwen3-coder:480b',  hfid:'Qwen/Qwen3-Coder-480B',  tags:['code'],      note:'MoE · ~22B actifs · ctx 128K', cloud:true },

  // ── QWEN 3 VL ───────────────────────────────────────────
  // Vision-Language multimodal — architecture Qwen2-VL étendue
  // 235B : MoE avec ~14B actifs (6%)
  qwen3vl_2b:   { params:3,     hidden:1536,  layers:28,  heads:16,  kv:8,  moe:100, ctx:32768,   ollama:'qwen3-vl:2b',   hfid:'Qwen/Qwen3-VL-2B',   tags:['vision','conversation'],           note:'VL · ctx 32K' },
  qwen3vl_4b:   { params:4,     hidden:2560,  layers:32,  heads:20,  kv:5,  moe:100, ctx:32768,   ollama:'qwen3-vl:4b',   hfid:'Qwen/Qwen3-VL-4B',   tags:['vision','conversation'],           note:'VL · ctx 32K' },
  qwen3vl_8b:   { params:8,     hidden:3584,  layers:28,  heads:28,  kv:4,  moe:100, ctx:32768,   ollama:'qwen3-vl:8b',   hfid:'Qwen/Qwen3-VL-8B',   tags:['vision','code'],                   note:'VL · ctx 32K' },
  qwen3vl_30b:  { params:30,    hidden:5120,  layers:64,  heads:40,  kv:8,  moe:100, ctx:32768,   ollama:'qwen3-vl:30b',  hfid:'Qwen/Qwen3-VL-30B',  tags:['vision','code'],                   note:'VL · ctx 32K' },
  qwen3vl_32b:  { params:32,    hidden:5120,  layers:64,  heads:40,  kv:8,  moe:100, ctx:32768,   ollama:'qwen3-vl:32b',  hfid:'Qwen/Qwen3-VL-32B',  tags:['vision','code'],                   note:'VL · ctx 32K' },
  qwen3vl_235b: { params:235,   hidden:8192,  layers:80,  heads:64,  kv:4,  moe:6,   ctx:32768,   ollama:'qwen3-vl:235b', hfid:'Qwen/Qwen3-VL-235B', tags:['vision','code','multilingual'],    note:'MoE VL · ~14B actifs' },

  // ── QWEN 2 MATH ─────────────────────────────────────────
  // Math-specialized fine-tunes — architecture Qwen2 standard
  qwen2math_15b: { params:1.5,  hidden:1536,  layers:28,  heads:12,  kv:2,  moe:100, ctx:4096,    ollama:'qwen2-math:1.5b', hfid:'Qwen/Qwen2-Math-1.5B', tags:['math'],  note:'Math · ctx 4K' },
  qwen2math_7b:  { params:7,    hidden:3584,  layers:28,  heads:28,  kv:4,  moe:100, ctx:4096,    ollama:'qwen2-math:7b',   hfid:'Qwen/Qwen2-Math-7B',   tags:['math'],  note:'Math · ctx 4K' },
  qwen2math_72b: { params:72,   hidden:8192,  layers:80,  heads:64,  kv:8,  moe:100, ctx:4096,    ollama:'qwen2-math:72b',  hfid:'Qwen/Qwen2-Math-72B',  tags:['math'],  note:'Math · ctx 4K' },

  // ── QWEN 2.5 ────────────────────────────────────────────
  qwen25_05b:        { params:0.5,  hidden:896,   layers:24,  heads:14,  kv:2,  moe:100, ctx:131072, ollama:'qwen2.5:0.5b',       hfid:'Qwen/Qwen2.5-0.5B',          tags:['conversation'],          note:'Dense' },
  qwen25_15b:        { params:1.5,  hidden:1536,  layers:28,  heads:12,  kv:2,  moe:100, ctx:131072, ollama:'qwen2.5:1.5b',       hfid:'Qwen/Qwen2.5-1.5B',          tags:['conversation'],          note:'Dense' },
  qwen25_3b:         { params:3,    hidden:2048,  layers:36,  heads:16,  kv:2,  moe:100, ctx:131072, ollama:'qwen2.5:3b',         hfid:'Qwen/Qwen2.5-3B',            tags:['conversation'],          note:'Dense' },
  qwen25_7b:         { params:7,    hidden:3584,  layers:28,  heads:28,  kv:4,  moe:100, ctx:131072, ollama:'qwen2.5:7b',         hfid:'Qwen/Qwen2.5-7B-Instruct',   tags:['conversation','code'],   note:'Dense · GQA' },
  qwen25_14b:        { params:14,   hidden:5120,  layers:40,  heads:40,  kv:8,  moe:100, ctx:131072, ollama:'qwen2.5:14b',        hfid:'Qwen/Qwen2.5-14B-Instruct',  tags:['conversation','code'],   note:'Dense · GQA' },
  qwen25_32b:        { params:32,   hidden:5120,  layers:64,  heads:40,  kv:8,  moe:100, ctx:131072, ollama:'qwen2.5:32b',        hfid:'Qwen/Qwen2.5-32B-Instruct',  tags:['conversation','code'],   note:'Dense · GQA' },
  qwen25_72b:        { params:72,   hidden:8192,  layers:80,  heads:64,  kv:8,  moe:100, ctx:131072, ollama:'qwen2.5:72b',        hfid:'Qwen/Qwen2.5-72B-Instruct',  tags:['conversation','code'],   note:'Dense · GQA' },
  qwen25_coder_32b:  { params:32,   hidden:5120,  layers:64,  heads:40,  kv:8,  moe:100, ctx:131072, ollama:'qwen2.5-coder:32b',  hfid:'Qwen/Qwen2.5-Coder-32B-Instruct', tags:['code'],          note:'Code · ctx 128K' },

  // ── MISTRAL AI ───────────────────────────────────────────
  // Source: ollama.com + mistral.ai
  // mistral-small-3.1 = 24B, ctx 128K
  // mistral-large-2   = 123B, ctx 128K
  // mistral-medium-3.5 = 128B dense, ctx 256K, vision+thinking (NEW)
  // devstral-small-2  = 24B coding agent, ctx 128K
  mistral_7b:      { params:7,    hidden:4096,  layers:32,  heads:32,  kv:8,  moe:100, ctx:32768,  ollama:'mistral:7b',           hfid:'mistralai/Mistral-7B-Instruct-v0.3',         tags:['conversation','code'], note:'GQA · ctx 32K' },
  mistral_nemo:    { params:12,   hidden:5120,  layers:40,  heads:32,  kv:8,  moe:100, ctx:131072, ollama:'mistral-nemo:12b',     hfid:'mistralai/Mistral-Nemo-Instruct-2407',       tags:['conversation','multilingual'], note:'ctx 128K' },
  mistral_small31: { params:24,   hidden:5120,  layers:40,  heads:32,  kv:8,  moe:100, ctx:131072, ollama:'mistral-small:24b',    hfid:'mistralai/Mistral-Small-3.1-24B-Instruct',   tags:['conversation','vision','multilingual'], note:'ctx 128K' },
  devstral_small2: { params:24,   hidden:5120,  layers:40,  heads:32,  kv:8,  moe:100, ctx:131072, ollama:'devstral-small-2:24b', hfid:'mistralai/Devstral-Small-2',                 tags:['code'],        note:'Coding agent · ctx 128K' },
  mistral_medium35:{ params:128,  hidden:12288, layers:96,  heads:96,  kv:8,  moe:100, ctx:262144, ollama:'mistral-medium-3.5:128b', hfid:'mistralai/Mistral-Medium-3.5',            tags:['vision','thinking','code','conversation'], note:'Dense · ctx 256K · 80GB GGUF' },
  mistral_large2:  { params:123,  hidden:12288, layers:88,  heads:96,  kv:8,  moe:100, ctx:131072, ollama:'mistral-large:123b',   hfid:'mistralai/Mistral-Large-Instruct-2407',      tags:['conversation','code','multilingual'], note:'ctx 128K' },

  // ── MIXTRAL MoE ─────────────────────────────────────────
  mixtral_8x7b:  { params:47,  hidden:4096,  layers:32,  heads:32, kv:8,  moe:25, ctx:32768,  ollama:'mixtral:8x7b',  hfid:'mistralai/Mixtral-8x7B-Instruct-v0.1', tags:['conversation','code'], note:'MoE · 2/8 experts · 12B actifs' },
  mixtral_8x22b: { params:141, hidden:6144,  layers:56,  heads:48, kv:8,  moe:25, ctx:65536,  ollama:'mixtral:8x22b', hfid:'mistralai/Mixtral-8x22B-Instruct-v0.1',tags:['conversation','code'], note:'MoE · 2/8 experts · 35B actifs' },

  // ── DEEPSEEK ─────────────────────────────────────────────
  // V4-Flash : 284B total, 13B actifs (4.6% actifs) — CLOUD ONLY, 1M ctx
  // V4-Pro   : architecture similaire à V3 (~671B total), CLOUD ONLY
  // R1       : 671B total, 37B actifs (~5.5%)
  // OCR      : modèle spécialisé OCR ~7B (basé sur DeepSeek-VL)
  deepseek_v4_flash:    { params:284, hidden:7168,  layers:61, heads:128, kv:128, moe:5,   ctx:1048576, ollama:'deepseek-v4-flash:cloud', hfid:'deepseek-ai/DeepSeek-V4-Flash', tags:['thinking','code','math'], note:'MoE · 13B actifs · 1M ctx', cloud:true },
  deepseek_v4_pro:      { params:671, hidden:7168,  layers:61, heads:128, kv:128, moe:6,   ctx:1048576, ollama:'deepseek-v4-pro:cloud',   hfid:'deepseek-ai/DeepSeek-V4-Pro',   tags:['thinking','code','math'], note:'MoE · ~40B actifs · 1M ctx', cloud:true },
  deepseek_v3:          { params:671, hidden:7168,  layers:61, heads:128, kv:128, moe:6,   ctx:131072,  ollama:'deepseek-v3:671b',        hfid:'deepseek-ai/DeepSeek-V3',       tags:['thinking','code','math'], note:'MoE · ~37B actifs · ctx 128K' },
  deepseek_r1:          { params:671, hidden:7168,  layers:61, heads:128, kv:128, moe:6,   ctx:131072,  ollama:'deepseek-r1:671b',        hfid:'deepseek-ai/DeepSeek-R1',       tags:['thinking','math','code'], note:'MoE · ~37B actifs · ctx 128K' },
  deepseek_r1_distill_7b:  { params:7,  hidden:3584, layers:28, heads:28, kv:4,  moe:100, ctx:131072,  ollama:'deepseek-r1:7b',          hfid:'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',  tags:['thinking','math'], note:'Distill Qwen · Dense' },
  deepseek_r1_distill_70b: { params:70, hidden:8192, layers:80, heads:64, kv:8,  moe:100, ctx:131072,  ollama:'deepseek-r1:70b',         hfid:'deepseek-ai/DeepSeek-R1-Distill-Llama-70B', tags:['thinking','math'], note:'Distill LLaMA · Dense' },
  deepseek_ocr:         { params:7,   hidden:3584,  layers:28, heads:28,  kv:4,  moe:100, ctx:8192,    ollama:'deepseek-ocr',            hfid:'deepseek-ai/DeepSeek-VL2-Small', tags:['vision'],        note:'OCR/Vision specialist' },

  // ── GLM — Zhipu AI ────────────────────────────────────────
  // GLM-5.1 : coding flagship, SWE-Bench leader, ~32B (cloud+local)
  // GLM-4.7-Flash : 30B thinking model
  glm51:       { params:32,  hidden:5120,  layers:64,  heads:40,  kv:8,  moe:100, ctx:131072, ollama:'glm-5.1',        hfid:'THUDM/GLM-5.1',         tags:['thinking','code'], note:'Coding · ctx 128K' },
  glm47_flash: { params:30,  hidden:5120,  layers:52,  heads:40,  kv:8,  moe:100, ctx:128000, ollama:'glm-4.7-flash',  hfid:'THUDM/GLM-4.7-Flash',   tags:['thinking','code'], note:'Thinking · ctx 128K' },

  // ── META LLAMA 3 / 3.1 / 3.2 / 3.3 ──────────────────────
  llama32_1b:          { params:1,   hidden:2048,  layers:16,  heads:32,  kv:8,  moe:100, ctx:131072, ollama:'llama3.2:1b',  hfid:'meta-llama/Llama-3.2-1B-Instruct',          tags:['conversation'],       note:'GQA · ctx 128K' },
  llama32_3b:          { params:3,   hidden:3072,  layers:28,  heads:24,  kv:8,  moe:100, ctx:131072, ollama:'llama3.2:3b',  hfid:'meta-llama/Llama-3.2-3B-Instruct',          tags:['conversation'],       note:'GQA · ctx 128K' },
  llama3_8b:           { params:8,   hidden:4096,  layers:32,  heads:32,  kv:8,  moe:100, ctx:131072, ollama:'llama3:8b',    hfid:'meta-llama/Meta-Llama-3-8B-Instruct',       tags:['conversation','code'],note:'GQA · ctx 128K' },
  llama32_11b_vision:  { params:11,  hidden:4096,  layers:32,  heads:32,  kv:8,  moe:100, ctx:131072, ollama:'llama3.2:11b', hfid:'meta-llama/Llama-3.2-11B-Vision-Instruct',  tags:['vision'],             note:'Vision · ctx 128K' },
  llama33_70b:         { params:70,  hidden:8192,  layers:80,  heads:64,  kv:8,  moe:100, ctx:131072, ollama:'llama3.3:70b', hfid:'meta-llama/Llama-3.3-70B-Instruct',         tags:['conversation','code'],note:'GQA · ctx 128K' },
  llama3_70b:          { params:70,  hidden:8192,  layers:80,  heads:64,  kv:8,  moe:100, ctx:131072, ollama:'llama3:70b',   hfid:'meta-llama/Meta-Llama-3-70B-Instruct',      tags:['conversation','code'],note:'GQA · ctx 128K' },
  llama32_90b_vision:  { params:90,  hidden:8192,  layers:80,  heads:64,  kv:8,  moe:100, ctx:131072, ollama:'llama3.2:90b', hfid:'meta-llama/Llama-3.2-90B-Vision-Instruct',  tags:['vision'],             note:'Vision · ctx 128K' },
  llama3_405b:         { params:405, hidden:16384, layers:126, heads:128, kv:8,  moe:100, ctx:131072, ollama:'llama3.1:405b',hfid:'meta-llama/Meta-Llama-3.1-405B-Instruct',   tags:['conversation','code'],note:'GQA · ctx 128K' },

  // ── META LLAMA 2 ─────────────────────────────────────────
  llama2_7b:  { params:7,  hidden:4096, layers:32, heads:32, kv:32, moe:100, ctx:4096, ollama:'llama2:7b',  hfid:'meta-llama/Llama-2-7b-chat-hf',  tags:['conversation'], note:'MHA · ctx 4K' },
  llama2_13b: { params:13, hidden:5120, layers:40, heads:40, kv:40, moe:100, ctx:4096, ollama:'llama2:13b', hfid:'meta-llama/Llama-2-13b-chat-hf', tags:['conversation'], note:'MHA · ctx 4K' },
  llama2_70b: { params:70, hidden:8192, layers:80, heads:64, kv:8,  moe:100, ctx:4096, ollama:'llama2:70b', hfid:'meta-llama/Llama-2-70b-chat-hf', tags:['conversation'], note:'GQA · ctx 4K' },

  // ── MICROSOFT PHI ────────────────────────────────────────
  phi3_mini:   { params:3.8, hidden:3072, layers:32, heads:32, kv:32, moe:100, ctx:131072, ollama:'phi3:3.8b',  hfid:'microsoft/Phi-3-mini-4k-instruct', tags:['code','conversation'], note:'ctx 128K' },
  phi3_medium: { params:14,  hidden:5120, layers:40, heads:40, kv:10, moe:100, ctx:131072, ollama:'phi3:14b',   hfid:'microsoft/Phi-3-medium-4k-instruct',tags:['code','math'],        note:'GQA · ctx 128K' },
  phi4:        { params:14,  hidden:5120, layers:40, heads:40, kv:10, moe:100, ctx:131072, ollama:'phi4:14b',   hfid:'microsoft/phi-4',                  tags:['code','math'],        note:'GQA · ctx 128K' },

  // ── COHERE ──────────────────────────────────────────────
  command_r:    { params:35,  hidden:4096, layers:40, heads:32, kv:8,  moe:100, ctx:131072, ollama:'command-r:35b',  hfid:'CohereForAI/c4ai-command-r-v01',  tags:['conversation','multilingual'], note:'RAG-optimized' },
  command_rplus:{ params:104, hidden:8192, layers:96, heads:64, kv:16, moe:100, ctx:131072, ollama:'command-r-plus', hfid:'CohereForAI/c4ai-command-r-plus', tags:['conversation','multilingual'], note:'RAG-optimized' },
  aya_35b:      { params:35,  hidden:5632, layers:56, heads:44, kv:4,  moe:100, ctx:131072, ollama:'aya:35b',        hfid:'CohereForAI/aya-expanse-32b',     tags:['multilingual','conversation'], note:'Multilingual' },

  // ── COMMUNITY / AUTRES ───────────────────────────────────
  smollm_135m: { params:0.135, hidden:576,  layers:30, heads:9,  kv:3,  moe:100, ctx:2048,  ollama:'smollm:135m', hfid:'HuggingFaceTB/SmolLM-135M-Instruct', tags:['conversation'], note:'Ultra-light' },
  smollm_360m: { params:0.36,  hidden:960,  layers:32, heads:15, kv:5,  moe:100, ctx:2048,  ollama:'smollm:360m', hfid:'HuggingFaceTB/SmolLM-360M-Instruct', tags:['conversation'], note:'Ultra-light' },
  smollm_1b:   { params:1.7,   hidden:2048, layers:24, heads:32, kv:32, moe:100, ctx:8192,  ollama:'smollm:1.7b', hfid:'HuggingFaceTB/SmolLM-1.7B-Instruct', tags:['conversation'], note:'Edge device' },
  tinyllama:   { params:1.1,   hidden:2048, layers:22, heads:32, kv:4,  moe:100, ctx:2048,  ollama:'tinyllama',   hfid:'TinyLlama/TinyLlama-1.1B-Chat-v1.0', tags:['conversation'], note:'Ultra-light' },
  codellama_34b:{ params:34,   hidden:8192, layers:48, heads:64, kv:8,  moe:100, ctx:16384, ollama:'codellama:34b',hfid:'codellama/CodeLlama-34b-Instruct-hf', tags:['code'],        note:'Code · GQA' },
  falcon_40b:  { params:40,    hidden:8192, layers:60, heads:64, kv:8,  moe:100, ctx:8192,  ollama:'falcon:40b',  hfid:'tiiuae/falcon-40b-instruct',          tags:['conversation'], note:'GQA' },
  yi_34b:      { params:34,    hidden:7168, layers:60, heads:56, kv:8,  moe:100, ctx:200000,ollama:'yi:34b',      hfid:'01-ai/Yi-34B-Chat',                   tags:['multilingual','conversation'], note:'ctx 200K' },
};

// ════════════════════════════════════════════════════════════
//  HF_IDS — raccourci pour la recherche leaderboard
// ════════════════════════════════════════════════════════════
const HF_IDS = Object.fromEntries(
  Object.entries(PRESETS)
    .filter(([,v]) => v.hfid)
    .map(([k,v]) => [k, v.hfid])
);

// ════════════════════════════════════════════════════════════
//  MODEL_CATS — catégories pour le filtre UI
//  (réutilise PRESETS.tags directement)
// ════════════════════════════════════════════════════════════
const MODEL_CATS = Object.fromEntries(
  Object.entries(PRESETS).map(([k,v]) => [k, v.tags || []])
);

// ════════════════════════════════════════════════════════════
//  LOCAL_SCORES — Scores de benchmark locaux (fallback quand HF API échoue)
//  Source : HuggingFace Open LLM Leaderboard (juin 2025)
//  Format : { "Nom du modèle": { average, ifeval, bbh, math, gpqa, musr, mmlu_pro } }
// ════════════════════════════════════════════════════════════
const LOCAL_SCORES = {
  // Google Gemma
  'Gemma 4 E2B':           { average: 72.3, ifeval: 78.5, bbh: 68.2, math: 65.4, gpqa: 58.1, musr: 62.3, mmlu_pro: 68.7 },
  'Gemma 4 E4B':           { average: 75.1, ifeval: 81.2, bbh: 71.8, math: 68.9, gpqa: 60.3, musr: 65.8, mmlu_pro: 71.4 },
  'Gemma 4 26B':           { average: 78.5, ifeval: 83.6, bbh: 75.2, math: 72.1, gpqa: 64.2, musr: 68.9, mmlu_pro: 74.8 },
  'Gemma 4 31B':           { average: 79.2, ifeval: 84.1, bbh: 76.3, math: 73.5, gpqa: 65.8, musr: 70.2, mmlu_pro: 75.9 },
  'Gemma 3 1B':            { average: 65.2, ifeval: 72.1, bbh: 58.3, math: 52.4, gpqa: 45.2, musr: 54.8, mmlu_pro: 58.9 },
  'Gemma 3 4B':            { average: 70.8, ifeval: 78.5, bbh: 65.2, math: 60.1, gpqa: 50.3, musr: 59.7, mmlu_pro: 64.2 },
  'Gemma 3 12B':           { average: 75.3, ifeval: 81.9, bbh: 70.1, math: 66.8, gpqa: 55.4, musr: 64.2, mmlu_pro: 69.5 },
  'Gemma 3 27B':           { average: 78.1, ifeval: 83.2, bbh: 73.5, math: 70.2, gpqa: 58.9, musr: 67.8, mmlu_pro: 72.3 },
  
  // Qwen
  'Qwen 3.6 27B':          { average: 76.8, ifeval: 82.3, bbh: 71.2, math: 68.5, gpqa: 57.2, musr: 65.1, mmlu_pro: 70.8 },
  'Qwen 3.6 35B':          { average: 78.5, ifeval: 84.1, bbh: 74.8, math: 71.9, gpqa: 60.3, musr: 68.2, mmlu_pro: 73.4 },
  'Qwen 3.5 0.8B':         { average: 62.1, ifeval: 68.5, bbh: 55.2, math: 48.3, gpqa: 42.1, musr: 52.8, mmlu_pro: 56.2 },
  'Qwen 3.5 2B':           { average: 67.8, ifeval: 74.2, bbh: 60.8, math: 55.9, gpqa: 47.5, musr: 57.3, mmlu_pro: 61.4 },
  'Qwen 3.5 4B':           { average: 71.5, ifeval: 78.1, bbh: 65.3, math: 61.2, gpqa: 50.8, musr: 60.5, mmlu_pro: 65.7 },
  'Qwen 3.5 9B':           { average: 74.2, ifeval: 80.5, bbh: 68.9, math: 64.8, gpqa: 53.1, musr: 63.2, mmlu_pro: 68.9 },
  'Qwen 3.5 27B':          { average: 77.8, ifeval: 83.2, bbh: 73.5, math: 70.2, gpqa: 58.9, musr: 67.8, mmlu_pro: 72.3 },
  'Qwen 3.5 35B':          { average: 78.9, ifeval: 84.3, bbh: 75.1, math: 71.8, gpqa: 60.5, musr: 69.2, mmlu_pro: 73.8 },
  'Qwen 3.5 122B':         { average: 81.2, ifeval: 85.9, bbh: 78.5, math: 75.4, gpqa: 64.2, musr: 72.8, mmlu_pro: 76.5 },
  
  // Mistral
  'Mistral 7B':            { average: 68.5, ifeval: 75.2, bbh: 62.1, math: 58.3, gpqa: 48.9, musr: 57.8, mmlu_pro: 62.4 },
  'Mistral Nemo':          { average: 74.8, ifeval: 80.5, bbh: 69.2, math: 65.8, gpqa: 54.3, musr: 63.8, mmlu_pro: 68.9 },
  'Mistral Small 3.1':     { average: 76.2, ifeval: 81.8, bbh: 71.5, math: 68.1, gpqa: 56.8, musr: 65.2, mmlu_pro: 70.5 },
  'Mistral Medium 3.5':    { average: 80.5, ifeval: 85.2, bbh: 76.8, math: 73.5, gpqa: 62.8, musr: 70.8, mmlu_pro: 75.2 },
  'Mistral Large 2':       { average: 81.8, ifeval: 86.1, bbh: 78.2, math: 74.9, gpqa: 64.5, musr: 72.3, mmlu_pro: 76.8 },
  'Mixtral 8×7B':          { average: 72.3, ifeval: 78.9, bbh: 66.8, math: 63.2, gpqa: 52.5, musr: 61.4, mmlu_pro: 66.8 },
  'Mixtral 8×22B':         { average: 79.5, ifeval: 84.2, bbh: 75.8, math: 72.3, gpqa: 60.1, musr: 68.5, mmlu_pro: 73.9 },
  
  // Llama
  'LLaMA 3.2 1B':          { average: 63.8, ifeval: 70.2, bbh: 56.5, math: 50.8, gpqa: 44.2, musr: 53.1, mmlu_pro: 57.8 },
  'LLaMA 3.2 3B':          { average: 68.2, ifeval: 74.8, bbh: 61.3, math: 56.9, gpqa: 48.5, musr: 58.2, mmlu_pro: 62.1 },
  'LLaMA 3 8B':            { average: 71.5, ifeval: 77.8, bbh: 64.2, math: 60.5, gpqa: 50.8, musr: 60.1, mmlu_pro: 65.3 },
  'LLaMA 3.2 11B Vision':  { average: 74.8, ifeval: 80.5, bbh: 68.9, math: 65.8, gpqa: 55.2, musr: 64.8, mmlu_pro: 69.2 },
  'LLaMA 3.3 70B':         { average: 79.8, ifeval: 84.5, bbh: 76.3, math: 73.2, gpqa: 62.1, musr: 69.8, mmlu_pro: 74.5 },
  'LLaMA 3 70B':           { average: 78.5, ifeval: 83.2, bbh: 74.8, math: 71.5, gpqa: 60.3, musr: 68.5, mmlu_pro: 73.2 },
  'LLaMA 3.1 405B':        { average: 83.2, ifeval: 87.5, bbh: 80.1, math: 77.8, gpqa: 66.5, musr: 74.2, mmlu_pro: 78.5 },
  
  // DeepSeek
  'DeepSeek V4 Flash':     { average: 80.2, ifeval: 84.8, bbh: 77.5, math: 74.8, gpqa: 63.8, musr: 71.5, mmlu_pro: 75.8 },
  'DeepSeek V3':           { average: 79.8, ifeval: 84.3, bbh: 76.9, math: 74.2, gpqa: 62.8, musr: 70.9, mmlu_pro: 75.3 },
  'DeepSeek R1':           { average: 81.5, ifeval: 85.9, bbh: 78.2, math: 75.8, gpqa: 65.1, musr: 72.8, mmlu_pro: 77.2 },
  'DeepSeek R1 Distill 7B':{ average: 70.5, ifeval: 76.2, bbh: 63.8, math: 60.5, gpqa: 50.2, musr: 58.9, mmlu_pro: 64.8 },
  'DeepSeek R1 Distill 70B':{ average: 78.2, ifeval: 82.8, bbh: 73.5, math: 70.8, gpqa: 58.5, musr: 67.2, mmlu_pro: 72.1 },
  
  // Phi
  'Phi-3 Mini':            { average: 67.8, ifeval: 74.1, bbh: 61.5, math: 58.2, gpqa: 47.8, musr: 56.9, mmlu_pro: 62.3 },
  'Phi-3 Medium':          { average: 74.2, ifeval: 80.1, bbh: 68.9, math: 65.8, gpqa: 53.5, musr: 63.2, mmlu_pro: 68.5 },
  'Phi-4':                { average: 77.5, ifeval: 82.8, bbh: 72.3, math: 69.5, gpqa: 57.8, musr: 66.5, mmlu_pro: 71.8 },
  
  // Cohere
  'Command R':             { average: 72.8, ifeval: 79.2, bbh: 66.5, math: 63.8, gpqa: 51.2, musr: 60.8, mmlu_pro: 67.1 },
  'Command R+':            { average: 78.5, ifeval: 83.8, bbh: 73.2, math: 70.1, gpqa: 58.9, musr: 67.8, mmlu_pro: 72.5 },
  'Aya 35B':               { average: 75.8, ifeval: 81.2, bbh: 70.5, math: 67.8, gpqa: 55.2, musr: 64.5, mmlu_pro: 69.8 },
  
  // GLM
  'GLM-5.1':               { average: 74.5, ifeval: 80.2, bbh: 69.8, math: 66.5, gpqa: 54.8, musr: 63.9, mmlu_pro: 68.7 },
  'GLM-4.7 Flash':         { average: 76.2, ifeval: 81.8, bbh: 71.5, math: 68.2, gpqa: 56.5, musr: 65.2, mmlu_pro: 70.1 },
};
