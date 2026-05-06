/**
 * LLM RAM/VRAM Calculator - Tests Unitaires
 * Exécuter avec : node js/tests.js
 */

// ============================================================================
// MOCK DES CONSTANTES ET FONCTIONS DEPENDANTES DU DOM
// ============================================================================

// Mock des frameworks
const FRAMEWORK_OH = {
  llamacpp:  { base:0.3,  scale:0.008, label:'llama.cpp / MLX' },
  ollama:    { base:0.5,  scale:0.010, label:'Ollama'          },
  vllm:      { base:1.5,  scale:0.015, label:'vLLM'            },
  tgi:       { base:1.2,  scale:0.012, label:'HF TGI'          },
  pytorch:   { base:2.0,  scale:0.018, label:'PyTorch/HF'      },
  exllamav2: { base:0.4,  scale:0.009, label:'ExLlamaV2'       },
};

// État des optimisations - réinitialisé pour chaque test
let optimState = { paged:false, fp8kv:false, int8kv:false, flashattn:false };

// ============================================================================
// FONCTIONS A TESTER (copiées depuis app.js)
// ============================================================================

function getOptimFactor() {
  let kv = 1.0, oh = 1.0;
  if (optimState.paged)   kv *= 0.75;
  if (optimState.fp8kv || optimState.int8kv) kv *= 0.5;
  if (optimState.flashattn) oh *= 0.7;
  return { kv, oh };
}

function calcModelWeight(paramsB, quant) {
  // Poids du modèle : (params_B × 10⁹ × quant_bits/8) / 1e9
  // NOTE: Le code actuel ne prend PAS en compte moe% pour le poids !
  // Tous les poids sont chargés même pour les modèles MoE
  return (paramsB * 1e9 * quant / 8) / 1e9;
}

function calcKVCache(layers, kvH, hidden, heads, ctx, batch, kvBits, optimFactor = 1.0) {
  // KV Cache : 2 × layers × kv_heads × (hidden/heads) × ctx × batch × (kvBits/8) / 1e9 × optim_factor
  const headDim = hidden / heads;
  return (2 * layers * kvH * headDim * ctx * batch * (kvBits / 8) / 1e9) * optimFactor;
}

function calcOverhead(fwKey, paramsB, optimFactor = 1.0) {
  const fw = FRAMEWORK_OH[fwKey] || FRAMEWORK_OH.vllm;
  return (fw.base + fw.scale * paramsB) * optimFactor;
}

function calcTotal(paramsB, hidden, layers, heads, kvH, quant, ctx, batch, kvBits, fwKey, optimFactorKV = 1.0, optimFactorOH = 1.0, mode = 'inference', trainMult = 1) {
  const wGB = calcModelWeight(paramsB, quant);
  const kvGB = calcKVCache(layers, kvH, hidden, heads, ctx, batch, kvBits, optimFactorKV);
  const ohGB = calcOverhead(fwKey, paramsB, optimFactorOH);
  
  let wGB_final = wGB;
  if (mode === 'training') wGB_final = wGB * trainMult;
  
  return wGB_final + kvGB + ohGB;
}

// ============================================================================
// HELPERS DE TEST
// ============================================================================

let passCount = 0;
let failCount = 0;

function assert(condition, testName, message = '') {
  if (condition) {
    passCount++;
    console.log(`✅ PASS: ${testName}`);
  } else {
    failCount++;
    console.log(`❌ FAIL: ${testName}` + (message ? ` - ${message}` : ''));
  }
}

function assertEqual(actual, expected, testName, tolerance = 0.01) {
  if (typeof expected === 'string') {
    if (actual === expected) {
      passCount++;
      console.log(`✅ PASS: ${testName}`);
    } else {
      failCount++;
      console.log(`❌ FAIL: ${testName} - Expected "${expected}", got "${actual}"`);
    }
  } else {
    const diff = Math.abs(actual - expected);
    if (diff <= tolerance) {
      passCount++;
      console.log(`✅ PASS: ${testName}`);
    } else {
      failCount++;
      console.log(`❌ FAIL: ${testName} - Expected ${expected.toFixed(4)}, got ${actual.toFixed(4)} (diff: ${diff.toFixed(4)})`);
    }
  }
}

function runTests() {
  console.log('\n========================================');
  console.log('LLM RAM/VRAM Calculator - Tests Unitaires');
  console.log('========================================\n');
  
  // ==========================================================================
  // TESTS: calcModelWeight
  // ==========================================================================
  console.log('--- Tests: calcModelWeight ---');
  
  // Llama 3 8B en INT4 (dense)
  // Formule : (8 * 1e9 * 4 / 8) / 1e9 = 4 GB
  assertEqual(calcModelWeight(8, 4), 4.0, 'Llama 3 8B INT4 = 4 GB');
  
  // Llama 3 8B en FP16 (dense)
  // Formule : (8 * 1e9 * 16 / 8) / 1e9 = 16 GB
  assertEqual(calcModelWeight(8, 16), 16.0, 'Llama 3 8B FP16 = 16 GB');
  
  // Llama 3 8B en FP32 (dense)
  // Formule : (8 * 1e9 * 32 / 8) / 1e9 = 32 GB
  assertEqual(calcModelWeight(8, 32), 32.0, 'Llama 3 8B FP32 = 32 GB');
  
  // Mixtral 8×7B (47B total) - NOTE: tous les poids sont chargés, pas juste les actifs
  // Formule : (47 * 1e9 * 4 / 8) / 1e9 = 23.5 GB
  assertEqual(calcModelWeight(47, 4), 23.5, 'Mixtral 8×7B INT4 = 23.5 GB (tous les poids)');
  
  // DeepSeek V4 Flash (284B total)
  // Formule : (284 * 1e9 * 4 / 8) / 1e9 = 142 GB
  assertEqual(calcModelWeight(284, 4), 142.0, 'DeepSeek V4 Flash INT4 = 142 GB (tous les poids)');
  
  // ==========================================================================
  // TESTS: calcKVCache
  // ==========================================================================
  console.log('\n--- Tests: calcKVCache ---');
  
  // Llama 3 8B : 32 layers, 32 heads, 8 KV heads, hidden=4096, ctx=4096, batch=1, FP16
  // headDim = 4096/32 = 128
  // KV = (2 * 32 * 8 * 128 * 4096 * 1 * 2) / 1e9 = 0.53687 GB
  assertEqual(calcKVCache(32, 8, 4096, 32, 4096, 1, 16), 0.536870912, 'Llama 3 8B KV Cache = 0.5369 GB', 0.001);
  
  // Llama 3 8B avec ctx=8192
  // KV = (2 * 32 * 8 * 128 * 8192 * 1 * 2) / 1e9 = 1.07374 GB
  assertEqual(calcKVCache(32, 8, 4096, 32, 8192, 1, 16), 1.073741824, 'Llama 3 8B KV Cache ctx=8192 = 1.0737 GB', 0.001);
  
  // Llama 3 8B avec batch=4
  // KV = (2 * 32 * 8 * 128 * 4096 * 4 * 2) / 1e9 = 2.14748 GB
  assertEqual(calcKVCache(32, 8, 4096, 32, 4096, 4, 16), 2.147483648, 'Llama 3 8B KV Cache batch=4 = 2.1475 GB', 0.001);
  
  // Llama 3 8B avec FP8 KV (kvBits=8)
  // KV = (2 * 32 * 8 * 128 * 4096 * 1 * 1) / 1e9 = 0.268435 GB
  assertEqual(calcKVCache(32, 8, 4096, 32, 4096, 1, 8), 0.268435456, 'Llama 3 8B KV Cache FP8 = 0.2684 GB', 0.001);
  
  // Avec optimisation PagedAttention (×0.75)
  assertEqual(calcKVCache(32, 8, 4096, 32, 4096, 1, 16, 0.75), 0.402653184, 'Llama 3 8B KV Cache + PagedAttention = 0.4027 GB', 0.001);
  
  // Avec optimisation FP8 KV (×0.5)
  assertEqual(calcKVCache(32, 8, 4096, 32, 4096, 1, 16, 0.5), 0.268435456, 'Llama 3 8B KV Cache + FP8 = 0.2684 GB', 0.001);
  
  // ==========================================================================
  // TESTS: calcOverhead
  // ==========================================================================
  console.log('\n--- Tests: calcOverhead ---');
  
  // llama.cpp avec Llama 3 8B
  // overhead = 0.3 + 0.008 * 8 = 0.364 GB
  assertEqual(calcOverhead('llamacpp', 8), 0.364, 'llama.cpp + 8B overhead = 0.364 GB');
  
  // vLLM avec Llama 3 8B
  // overhead = 1.5 + 0.015 * 8 = 1.62 GB
  assertEqual(calcOverhead('vllm', 8), 1.62, 'vLLM + 8B overhead = 1.62 GB');
  
  // PyTorch avec Llama 3 70B
  // overhead = 2.0 + 0.018 * 70 = 3.26 GB
  assertEqual(calcOverhead('pytorch', 70), 3.26, 'PyTorch + 70B overhead = 3.26 GB');
  
  // ==========================================================================
  // TESTS: getOptimFactor
  // ==========================================================================
  console.log('\n--- Tests: getOptimFactor ---');
  
  // Aucun optimisation
  optimState = { paged:false, fp8kv:false, int8kv:false, flashattn:false };
  const noOpt = getOptimFactor();
  assert(noOpt.kv === 1.0 && noOpt.oh === 1.0, 'No optimizations: kv=1.0, oh=1.0');
  
  // PagedAttention seulement
  optimState = { paged:true, fp8kv:false, int8kv:false, flashattn:false };
  const paged = getOptimFactor();
  assert(paged.kv === 0.75 && paged.oh === 1.0, 'PagedAttention: kv=0.75, oh=1.0');
  
  // FP8 KV seulement
  optimState = { paged:false, fp8kv:true, int8kv:false, flashattn:false };
  const fp8 = getOptimFactor();
  assert(fp8.kv === 0.5 && fp8.oh === 1.0, 'FP8 KV: kv=0.5, oh=1.0');
  
  // INT8 KV seulement
  optimState = { paged:false, fp8kv:false, int8kv:true, flashattn:false };
  const int8 = getOptimFactor();
  assert(int8.kv === 0.5 && int8.oh === 1.0, 'INT8 KV: kv=0.5, oh=1.0');
  
  // FlashAttention seulement
  optimState = { paged:false, fp8kv:false, int8kv:false, flashattn:true };
  const flash = getOptimFactor();
  assert(flash.kv === 1.0 && flash.oh === 0.7, 'FlashAttention: kv=1.0, oh=0.7');
  
  // FP8 KV + FlashAttention
  optimState = { paged:false, fp8kv:true, int8kv:false, flashattn:true };
  const fp8flash = getOptimFactor();
  assert(fp8flash.kv === 0.5 && fp8flash.oh === 0.7, 'FP8 KV + FlashAttention: kv=0.5, oh=0.7');
  
  // PagedAttention + FP8 KV
  optimState = { paged:true, fp8kv:true, int8kv:false, flashattn:false };
  const pagedfp8 = getOptimFactor();
  assert(pagedfp8.kv === 0.375 && pagedfp8.oh === 1.0, 'PagedAttention + FP8 KV: kv=0.375, oh=1.0');
  
  // ==========================================================================
  // TESTS: calcTotal (Intégration)
  // ==========================================================================
  console.log('\n--- Tests: calcTotal (Intégration) ---');
  
  // Llama 3 8B, INT4, ctx=4096, llama.cpp
  // Poids: 4 GB, KV: ~0.5369 GB, Overhead: 0.364 GB
  // Total: ~4.90 GB
  const ll3_8b = calcTotal(8, 4096, 32, 32, 8, 4, 4096, 1, 16, 'llamacpp');
  assert(ll3_8b >= 4.89 && ll3_8b <= 4.91, `Llama 3 8B INT4 total ~4.90 GB (got ${ll3_8b.toFixed(4)})`);
  
  // Llama 3 8B, FP16, ctx=4096, vLLM
  // Poids: 16 GB, KV: ~0.5369 GB, Overhead: 1.62 GB
  // Total: ~18.16 GB
  const ll3_8b_fp16 = calcTotal(8, 4096, 32, 32, 8, 16, 4096, 1, 16, 'vllm');
  assert(ll3_8b_fp16 >= 18.15 && ll3_8b_fp16 <= 18.17, `Llama 3 8B FP16 total ~18.16 GB (got ${ll3_8b_fp16.toFixed(4)})`);
  
  // Mixtral 8×7B, INT4, ctx=32768, llama.cpp
  // Poids: 23.5 GB, KV: (2 * 32 * 8 * 128 * 32768 * 1 * 2) / 1e9 = 4.295 GB
  // Overhead llamacpp: 0.3 + 0.008*47 = 0.676 GB
  // Total: ~28.47 GB
  const mx8x7b = calcTotal(47, 4096, 32, 32, 8, 4, 32768, 1, 16, 'llamacpp');
  assert(mx8x7b >= 28.46 && mx8x7b <= 28.48, `Mixtral 8×7B INT4 total ~28.47 GB (got ${mx8x7b.toFixed(4)})`);
  
  // ==========================================================================
  // TESTS: Mode Entraînement
  // ==========================================================================
  console.log('\n--- Tests: Mode Entraînement ---');
  
  // Llama 3 7B, FP16, LoRA (×5.5)
  // Poids: 14 GB * 5.5 = 77 GB, KV: ~0.5369 GB, Overhead: 2.0 + 0.018*7 = 2.126 GB
  // Total: ~79.66 GB
  const ll3_7b_lora = calcTotal(7, 4096, 32, 32, 8, 16, 4096, 1, 16, 'pytorch', 1, 1, 'training', 5.5);
  assert(ll3_7b_lora >= 79.65 && ll3_7b_lora <= 79.67, `Llama 3 7B FP16 LoRA total ~79.66 GB (got ${ll3_7b_lora.toFixed(4)})`);
  
  // Llama 3 7B, FP16, QLoRA (×3)
  // Poids: 14 GB * 3 = 42 GB
  // Total: ~44.66 GB
  const ll3_7b_qlora = calcTotal(7, 4096, 32, 32, 8, 16, 4096, 1, 16, 'pytorch', 1, 1, 'training', 3);
  assert(ll3_7b_qlora >= 44.65 && ll3_7b_qlora <= 44.67, `Llama 3 7B FP16 QLoRA total ~44.66 GB (got ${ll3_7b_qlora.toFixed(4)})`);
  
  // Llama 3 7B, FP16, Full FT (×8)
  // Poids: 14 GB * 8 = 112 GB
  // Total: ~114.66 GB
  const ll3_7b_fullft = calcTotal(7, 4096, 32, 32, 8, 16, 4096, 1, 16, 'pytorch', 1, 1, 'training', 8);
  assert(ll3_7b_fullft >= 114.65 && ll3_7b_fullft <= 114.67, `Llama 3 7B FP16 Full FT total ~114.66 GB (got ${ll3_7b_fullft.toFixed(4)})`);
  
  // ==========================================================================
  // TESTS: cleanForLeaderboard
  // ==========================================================================
  console.log('\n--- Tests: cleanForLeaderboard ---');
  
  function cleanForLeaderboard(name) {
    return name
      .replace(/[-_]?(fp8|fp16|fp32|bf16|int4|int8|w4a16|w8a16|awq|gptq|gguf|ggml|exl2|hqq|nf4|q[2468]_?[km]?|q\d)/gi, '')
      .replace(/[-_]?(instruct|chat|it|hf|turbo|plus|ultra|lite|mini|nano|base|v\d[\d.]*)/gi, '')
      .replace(/[-_]+[km]\d?/gi, '')  // Enlever _K, _M, _K_M, etc.
      .replace(/[-_\.]+$/g, '')
      .trim();
  }
  
  assertEqual(cleanForLeaderboard('Qwen3.5-4B-Instruct-FP8'), 'Qwen3.5-4B', 'Clean Qwen3.5-4B-Instruct-FP8');
  assertEqual(cleanForLeaderboard('Llama-3-8B-Instruct'), 'Llama-3-8B', 'Clean Llama-3-8B-Instruct');
  assertEqual(cleanForLeaderboard('Mistral-7B-v0.3-GGUF-Q4_K_M'), 'Mistral-7B', 'Clean Mistral-7B-v0.3-GGUF-Q4_K_M');
  assertEqual(cleanForLeaderboard('gemma-4-e2b-it'), 'gemma-4-e2b', 'Clean gemma-4-e2b-it');
  
  // ==========================================================================
  // SUMMARY
  // ==========================================================================
  console.log('\n========================================');
  console.log(`Résumé: ${passCount} ✅ PASS, ${failCount} ❌ FAIL`);
  console.log('========================================\n');
  
  if (failCount > 0) {
    process.exit(1);
  }
}

// Execute tests
runTests();
