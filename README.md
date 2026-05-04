# 🧠 LLM RAM/VRAM Calculator v5.2

> **Calculateur local IA** – Outil complet pour estimer la consommation mémoire (RAM/VRAM) des modèles de langage (LLM) en inférence ou entraînement.

---

## 📁 Arborescence du projet

```
Version-en-cours/
├── index.html               # Interface principale
├── css/
│   └── style.css            # Styles CSS
├── js/
│   └── app.js               # Logique JavaScript
├── models.js                # Base de données des modèles (40+ LLM)
└── README.md                # Ce fichier
```

---

## 📌 Résumé

**LLM RAM/VRAM Calculator v5.2** est un outil **100% local** (fonctionne dans le navigateur) conçu pour aider les développeurs, chercheurs et passionnés à :

- **Estimer** la consommation mémoire (RAM/VRAM) d'un LLM en **inférence** ou **entraînement**
- **Comparer** les modèles entre eux (poids, KV cache, overhead)
- **Optimiser** la configuration (quantification, frameworks, optimisations KV cache)
- **Vérifier** la compatibilité matérielle (Apple Silicon, NVIDIA, AMD)
- **Récupérer** les benchmarks depuis **HuggingFace Open LLM Leaderboard** ou **llm-stats.com**
- **Générer** des commandes `ollama run` pour un déploiement local

---

## ✨ Fonctionnalités clés

### 🔢 Calculs précis
- **Formules v4** corrigées pour GQA/MQA (Grouped-Query Attention / Multi-Query Attention)
- **Poids du modèle** : `(params_B × moe% × 10⁹ × quant_bits/8) / 1e9` → GB
- **KV Cache** : `2 × layers × kv_heads × (hidden/heads) × ctx × batch × (kvBits/8) / 1e9 × optim_factor`
- **Overhead dynamique** : `base_framework + 0.015 × params_B` (évolutif avec la taille du modèle)
- **Entraînement** : Multiplicateurs réalistes (Full FT ×8, LoRA ×5.5, QLoRA ×3)

### 🎯 Modes disponibles
| Mode | Description | Multiplicateur mémoire |
|------|-------------|----------------------|
| **⚡ Inférence** | Exécution standard du modèle | ×1 |
| **🔬 Entraînement** | Fine-tuning complet | ×8 (Full FT) / ×5.5 (LoRA) / ×3 (QLoRA) |

### 📊 Optimisations KV Cache (v4)
| Optimisation | Facteur KV Cache | Description |
|--------------|------------------|-------------|
| **PagedAttention** (vLLM) | ×0.75 | Gestion paginée de la mémoire |
| **Quantification KV FP8** | ×0.5 | Réduction de précision du KV cache |
| **Quantification KV INT8** | ×0.5 | Alternative à FP8 |
| **FlashAttention-2/3** | ×0.7 (overhead) | Réduction des activations intermédiaires |

### 🖥️ Frameworks supportés
| Framework | Overhead base (GB) | Scale par milliard de paramètres |
|-----------|-------------------|----------------------------------|
| llama.cpp / MLX | 0.3 | 0.008 |
| Ollama | 0.5 | 0.010 |
| vLLM | 1.5 | 0.015 |
| HuggingFace TGI | 1.2 | 0.012 |
| PyTorch / Transformers | 2.0 | 0.018 |
| ExLlamaV2 | 0.4 | 0.009 |

---

## 🚀 Utilisation

### 1️⃣ **Sélectionner un modèle**
- Choisir parmi **40+ modèles prédéfinis** (Gemma 4, Qwen 3.6, Llama 3.3, Mistral, DeepSeek, etc.)
- Ou utiliser le **mode manuel** pour saisir des paramètres personnalisés
- Ou **ajouter un modèle personnalisé** (stocké dans `localStorage`)

### 2️⃣ **Configurer les paramètres**
#### Mode Auto (recommandé)
- **Quantification** : INT4 (4 bits), INT8, FP16, FP32
- **Contexte** : Nombre de tokens (jusqu'à 1M pour certains modèles)
- **Framework** : llama.cpp, Ollama, vLLM, etc.

#### Mode Manuel (expert)
- **Paramètres** : Nombre de milliards de paramètres
- **Hidden size** : Taille des vecteurs internes
- **Num layers** : Nombre de couches
- **Num heads** : Nombre de têtes d'attention
- **KV heads** : Nombre de têtes KV (GQA si < num heads)
- **MoE actif** : Pourcentage d'experts actifs (modèles MoE)
- **Batch size** : Nombre de requêtes simultanées
- **Précision KV** : Bits pour le KV cache (FP16, FP8, INT8)

### 3️⃣ **Activer les optimisations**
- Cocher les optimisations KV cache utilisées (PagedAttention, FP8 KV, etc.)
- Le calcul s'ajuste automatiquement

### 4️⃣ **Consulter les résultats**
- **Poids du modèle** : Mémoire nécessaire pour les poids
- **KV Cache** : Mémoire pour le cache de contexte
- **Overhead runtime** : Mémoire du framework
- **Total RAM/VRAM** : **Somme des 3** (en GB et MiB)

### 5️⃣ **Vérifier la compatibilité matérielle**
- Filtrer par type de matériel : **Apple Silicon** (🍎), **NVIDIA** (🟢), **AMD** (🔴)
- Voir le **débit estimé** (tok/s) et l'**efficacité énergétique** (tok/s/W)
- **Recommandation automatique** : Meilleur choix selon l'usage (Général, Math, Code, Conversation)

### 6️⃣ **Comparer les modèles** (v5.1)
- Ajouter plusieurs modèles avec leurs configurations
- Visualiser la consommation mémoire côté à côté
- Graphique SVG dynamique

### 7️⃣ **Récupérer les benchmarks**
- **HuggingFace Open LLM Leaderboard** (gratuit)
- **llm-stats.com** (nécessite une clé API)
- Scores disponibles : Average, IFEval, BBH, MATH Lvl 5, GPQA, MuSR, MMLU-PRO
- **Ratio performance/ressource** : MMLU-PRO % / GB RAM/VRAM

---

## 📊 Modèles supportés (40+)

### 🟣 **Google DeepMind**
| Modèle | Paramètres | Contexte | Type | Ollama |
|--------|------------|----------|------|--------|
| Gemma 4 E2B | 13B (2B actifs) | 128K | MoE | `gemma4:e2b` |
| Gemma 4 E4B | 17B (4B actifs) | 128K | MoE | `gemma4:e4b` |
| Gemma 4 26B | 26B (4B actifs) | 256K | MoE | `gemma4:26b` |
| Gemma 4 31B | 31B | 256K | Dense | `gemma4:31b` |
| Gemma 3 1B | 1B | 32K | Dense | `gemma3:1b` |
| Gemma 3 4B | 4B | 128K | Dense | `gemma3:4b` |
| Gemma 3 12B | 12B | 128K | Dense | `gemma3:12b` |
| Gemma 3 27B | 27B | 128K | Dense | `gemma3:27b` |

### 🔵 **Alibaba (Qwen)**
| Modèle | Paramètres | Contexte | Type | Tags |
|--------|------------|----------|------|------|
| Qwen 3.6 27B | 27B | 256K | Dense | Thinking, Code, Conversation |
| Qwen 3.6 35B | 35B (3B actifs) | 256K | MoE A3B | Thinking, Code |
| Qwen 3.5 0.8B | 0.8B | 256K | Dense | Thinking, Vision, Conversation |
| Qwen 3.5 2B | 2B | 256K | Dense | Thinking, Vision, Conversation |
| Qwen 3.5 4B | 4B | 256K | Dense | Thinking, Vision, Code |
| Qwen 3.5 9B | 9B | 256K | Dense | Thinking, Vision, Code |
| Qwen 3.5 27B | 27B | 256K | Dense | Thinking, Vision, Code, Math |
| Qwen 3.5 35B | 35B | 256K | Dense | Thinking, Vision, Code, Math |
| Qwen 3.5 122B | 122B | 256K | Dense | Thinking, Vision, Code, Math |
| Qwen 3 Coder 30B | 30B | 128K | Dense | Code |
| Qwen 3 Coder 480B | 480B (~22B actifs) | 128K | MoE | Code, ☁️ Cloud-only |
| Qwen 3 VL 2B-235B | 2B-235B | 32K-32K | VL/MoE | Vision |
| Qwen 2.5 0.5B-72B | 0.5B-72B | 128K | Dense | Conversation, Code |
| Qwen 2 Math 1.5B-72B | 1.5B-72B | 4K | Dense | Math |

### 🟤 **Mistral AI**
| Modèle | Paramètres | Contexte | Type | Ollama |
|--------|------------|----------|------|--------|
| Mistral 7B | 7B | 32K | GQA | `mistral:7b` |
| Mistral Nemo | 12B | 128K | Dense | `mistral-nemo:12b` |
| Mistral Small 3.1 | 24B | 128K | Dense | `mistral-small:24b` |
| Devstral Small 2 | 24B | 128K | Dense | `devstral-small-2:24b` |
| Mistral Medium 3.5 | 128B | 256K | Dense | `mistral-medium-3.5:128b` |
| Mistral Large 2 | 123B | 128K | Dense | `mistral-large:123b` |
| Mixtral 8×7B | 47B (12B actifs) | 32K | MoE | `mixtral:8x7b` |
| Mixtral 8×22B | 141B (35B actifs) | 64K | MoE | `mixtral:8x22b` |

### 🔴 **DeepSeek AI**
| Modèle | Paramètres | Contexte | Type | Tags |
|--------|------------|----------|------|------|
| DeepSeek V4 Flash | 284B (13B actifs) | 1M | MoE | Thinking, Code, Math, ☁️ Cloud-only |
| DeepSeek V4 Pro | 671B (~40B actifs) | 1M | MoE | Thinking, Code, Math, ☁️ Cloud-only |
| DeepSeek V3 | 671B (~37B actifs) | 128K | MoE | Thinking, Code, Math |
| DeepSeek R1 | 671B (~37B actifs) | 128K | MoE | Thinking, Math, Code |
| DeepSeek R1 Distill 7B | 7B | 128K | Dense | Thinking, Math |
| DeepSeek R1 Distill 70B | 70B | 128K | Dense | Thinking, Math |
| DeepSeek OCR | 7B | 8K | Dense | Vision |

### 🟠 **Meta (LLaMA)**
| Modèle | Paramètres | Contexte | Type | Ollama |
|--------|------------|----------|------|--------|
| LLaMA 3.2 1B | 1B | 128K | GQA | `llama3.2:1b` |
| LLaMA 3.2 3B | 3B | 128K | GQA | `llama3.2:3b` |
| LLaMA 3 8B | 8B | 128K | GQA | `llama3:8b` |
| LLaMA 3.2 11B Vision | 11B | 128K | GQA | `llama3.2:11b` |
| LLaMA 3.3 70B | 70B | 128K | GQA | `llama3.3:70b` |
| LLaMA 3 70B | 70B | 128K | GQA | `llama3:70b` |
| LLaMA 3.2 90B Vision | 90B | 128K | GQA | `llama3.2:90b` |
| LLaMA 3.1 405B | 405B | 128K | GQA | `llama3.1:405b` |
| LLaMA 2 7B-70B | 7B-70B | 4K | MHA/GQA | `llama2:7b`, `llama2:13b`, `llama2:70b` |

### 🟡 **Microsoft (Phi)**
| Modèle | Paramètres | Contexte | Type | Ollama |
|--------|------------|----------|------|--------|
| Phi-3 Mini | 3.8B | 128K | Dense | `phi3:3.8b` |
| Phi-3 Medium | 14B | 128K | GQA | `phi3:14b` |
| Phi-4 | 14B | 128K | GQA | `phi4:14b` |

### ⚫ **Cohere**
| Modèle | Paramètres | Contexte | Type | Ollama |
|--------|------------|----------|------|--------|
| Command R | 35B | 128K | Dense | `command-r:35b` |
| Command R+ | 104B | 128K | Dense | `command-r-plus` |
| Aya 35B | 35B | 128K | Dense | `aya:35b` |

### 🟢 **Zhipu AI (GLM)**
| Modèle | Paramètres | Contexte | Type | Ollama |
|--------|------------|----------|------|--------|
| GLM-5.1 | 32B | 128K | Dense | `glm-5.1` |
| GLM-4.7 Flash | 30B | 128K | Dense | `glm-4.7-flash` |

### ⚪ **Autres**
| Modèle | Paramètres | Contexte | Type | Ollama |
|--------|------------|----------|------|--------|
| SmolLM 135M-1B | 0.135B-1.7B | 2K-8K | Dense | `smollm:135m`, `smollm:360m`, `smollm:1.7b` |
| TinyLlama | 1.1B | 2K | Dense | `tinyllama` |
| CodeLlama 34B | 34B | 16K | GQA | `codellama:34b` |
| Falcon 40B | 40B | 8K | GQA | `falcon:40b` |
| Yi 34B | 34B | 200K | Dense | `yi:34b` |

---

## 🖥️ Matériel supporté

### 🍎 **Apple Silicon** (Mémoire unifiée RAM=VRAM)
| Modèle | VRAM | Bande passante (GB/s) | TDP (W) |
|--------|------|----------------------|---------|
| Mac mini M4 | 16 GB | 120 | 25 |
| MacBook Pro M4 Pro | 24 GB | 273 | 30 |
| Mac M4 Pro | 48 GB | 273 | 40 |
| Mac M4 Max | 64 GB | 410 | 55 |
| Mac M4 Max | 128 GB | 410 | 55 |
| Mac M3 Pro | 36 GB | 150 | 30 |
| Mac M3 Max | 96 GB | 300 | 50 |
| Mac M2 Ultra | 192 GB | 800 | 180 |

### 🟢 **NVIDIA** (VRAM dédiée)
| Modèle | VRAM | Bande passante (GB/s) | TDP (W) |
|--------|------|----------------------|---------|
| RTX 4060 Ti | 16 GB | 288 | 165 |
| RTX 4080 | 16 GB | 716 | 320 |
| RTX 4090 | 24 GB | 1008 | 450 |
| RTX 5090 | 32 GB | 1792 | 575 |
| A100 SXM | 40 GB | 1555 | 400 |
| A100 SXM | 80 GB | 2039 | 400 |
| H100 SXM | 80 GB | 3350 | 700 |
| H200 | 141 GB | 4800 | 700 |
| B200 | 192 GB | 8000 | 1000 |

### 🔴 **AMD** (VRAM dédiée)
| Modèle | VRAM | Bande passante (GB/s) | TDP (W) |
|--------|------|----------------------|---------|
| RX 7900 XTX | 24 GB | 960 | 355 |
| AMD MI300X | 192 GB | 5300 | 750 |

### 📱 **Smartphones** (Mémoire partagée RAM)
| Modèle | VRAM | Bande passante (GB/s) | TDP (W) |
|--------|------|----------------------|---------|
| iPhone 15 Pro | 8 GB | 50 | 15 |
| iPhone 15 Pro Max | 8 GB | 50 | 18 |
| iPhone 16 Pro | 12 GB | 60 | 18 |
| Samsung Galaxy S24 Ultra | 12 GB | 70 | 15 |
| Google Pixel 8 Pro | 12 GB | 65 | 12 |
| Samsung Galaxy S25 Ultra | 16 GB | 80 | 18 |

> ⚠️ **Note** : Les smartphones ont une mémoire **partagée** (RAM = VRAM). Les performances d'inférence sont limitées par la RAM disponible et la bande passante mémoire. Seuls les petits modèles (< 8B en INT4) sont réalistes sur smartphone.

---

## 🎨 Interface Utilisateur

### 📱 **Design**
- **Thème sombre** optimisé pour le confort visuel
- **Responsive** : Adapté aux écrans mobiles, tablettes et desktop
- **Animations fluides** : Transitions et effets visuels (grille animée, compteurs)
- **Polices** : JetBrains Mono (code), DM Sans (UI), Syne (titres)

### 🎯 **Sections principales**
1. **En-tête** : Titre, version, bascule Inférence/Entraînement
2. **Configuration du modèle** : Sélection, quantification, contexte, framework
3. **Configuration d'entraînement** (si mode Entraînement) : Méthode (Full FT, LoRA, QLoRA)
4. **Optimisations KV Cache** : PagedAttention, FP8 KV, INT8 KV, FlashAttention
5. **Résultats** : Poids, KV Cache, Overhead, Total RAM/VRAM
6. **Benchmarks** : Récupération depuis HF Leaderboard ou llm-stats.com
7. **Compatibilité matérielle** : Filtres par type, recommandations
8. **Comparateur de modèles** : Ajout de plusieurs modèles pour comparaison
9. **Formules** : Explications mathématiques des calculs

---

## 🔧 Fonctionnalités techniques

### 📦 **Gestion des modèles**
- **Base de données centralisée** dans `models.js`
- **Structure par familles** (Google, Alibaba, Mistral, etc.)
- **Catégorisation** : Thinking, Code, Math, Vision, Conversation, Multilingual
- **Stockage local** : Modèles personnalisés sauvegardés dans `localStorage`

### ⚡ **Calculs en temps réel**
- **Écouteurs d'événements** sur tous les champs de saisie
- **Mise à jour instantanée** des résultats
- **Animations CSS** pour les changements de valeurs

### 🌐 **Intégrations API**
- **HuggingFace Open LLM Leaderboard** (v2)
  - Recherche par ID modèle
  - Stratégie de fallback en 4 étapes (nom brut → nom nettoyé → nom de base → API Hub)
  - Récupération des scores : Average, IFEval, BBH, MATH Lvl 5, GPQA, MuSR, MMLU-PRO
- **llm-stats.com** (nécessite clé API)
  - Alternative au leaderboard HF
  - Scores similaires + métriques supplémentaires

### 🎨 **Système de tooltips**
- **Info-bulles contextuelles** sur tous les champs
- **Explications détaillées** des concepts (GQA, MoE, KV Cache, etc.)
- **Design élégant** avec animations

### 📊 **Visualisations**
- **Graphique SVG** pour le comparateur de modèles
- **Barres de progression** pour la compatibilité matérielle
- **Badges colorés** pour les statuts (Compatible, Limite, Insuffisant)

---

## 📈 Exemples d'utilisation

### 🔹 **Cas 1 : Déploiement local de Llama 3 8B en INT4**
1. Sélectionner `Llama 3 8B` dans la liste des modèles
2. Choisir `INT4` (4 bits) pour la quantification
3. Définir le contexte à `4096` tokens
4. Sélectionner `llama.cpp` comme framework
5. **Résultat** : ~4.3 GB de RAM/VRAM nécessaires
6. **Matériel compatible** : RTX 4060 Ti (16 GB), Mac M4 Pro (24 GB), etc.
7. **Commande Ollama** : `ollama run llama3:8b` (affichée automatiquement)

### 🔹 **Cas 2 : Entraînement LoRA de Mistral 7B**
1. Basculer en mode **Entraînement**
2. Sélectionner `Mistral 7B`
3. Choisir `FP16` (16 bits) pour la quantification
4. Sélectionner `LoRA` comme méthode d'entraînement (×5.5)
5. **Résultat** : ~77 GB de VRAM nécessaires (7B × 5.5 × 2 + KV Cache + Overhead)
6. **Matériel requis** : A100 80 GB, H100 80 GB, etc.

### 🔹 **Cas 3 : Comparaison Qwen 3.5 4B vs Llama 3 8B**
1. Aller dans la section **Comparateur de modèles**
2. Ajouter `Qwen 3.5 4B` avec INT4
3. Ajouter `Llama 3 8B` avec INT4
4. **Résultat** : Graphique comparatif montrant que Qwen 3.5 4B consomme ~2.2 GB vs ~4.3 GB pour Llama 3 8B

### 🔹 **Cas 4 : Benchmark de Gemma 3 4B**
1. Sélectionner `Gemma 3 4B`
2. Cliquer sur **Chercher benchmarks**
3. **Résultat** : Scores du Open LLM Leaderboard (Average, IFEval, BBH, etc.)
4. **Ratio performance/ressource** : ~15-20 pts MMLU-PRO / GB

---

## 🛠️ Personnalisation

### ✏️ **Ajouter un modèle personnalisé**
1. Cliquer sur **➕ Ajouter un modèle**
2. Remplir les champs :
   - Nom affiché
   - ID HuggingFace (optionnel)
   - Paramètres (B)
   - Hidden size
   - Num layers
   - Num heads
   - KV heads
   - MoE actif (%) (100 pour les modèles denses)
3. Cliquer sur **💾 Sauvegarder**
4. Le modèle apparaît dans la liste **Mes modèles sauvegardés**

### 🔧 **Mode Manuel**
Pour les modèles non listés ou les configurations avancées :
1. Sélectionner **🔧 Saisie manuelle / Autre modèle**
2. Basculer en mode **Manuel** (bouton en haut)
3. Remplir tous les paramètres architecturaux
4. Les résultats sont calculés en temps réel

---

## 📊 Formules détaillées (v4)

### 🔹 **Poids du modèle**
```
Poids (GB) = (params_B × moe_pct/100 × 10⁹ × quant_bits/8) / 1e9
```
- `params_B` : Nombre de milliards de paramètres
- `moe_pct` : Pourcentage de paramètres actifs (100 pour les modèles denses, <100 pour MoE)
- `quant_bits` : Bits par poids (4 pour INT4, 8 pour INT8, 16 pour FP16, etc.)

**Exemple** : Llama 3 8B en INT4 → `(8 × 100/100 × 1e9 × 4/8) / 1e9 = 4 GB`

### 🔹 **KV Cache (GQA-correct)**
```
KV Cache (GB) = 2 × layers × kv_heads × (hidden/heads) × ctx × batch × (kvBits/8) / 1e9 × optim_factor
```
- `layers` : Nombre de couches
- `kv_heads` : Nombre de têtes KV (peut être < heads pour GQA)
- `hidden` : Hidden size (dimension des vecteurs)
- `heads` : Nombre total de têtes d'attention
- `ctx` : Longueur du contexte en tokens
- `batch` : Taille du batch
- `kvBits` : Bits par élément KV (16 pour FP16, 8 pour FP8/INT8)
- `optim_factor` : Facteur de réduction des optimisations (0.75 pour PagedAttention, 0.5 pour FP8 KV, etc.)

**Exemple** : Llama 3 8B (32 layers, 32 heads, 8 KV heads, hidden=4096) avec ctx=4096, batch=1, FP16 :
`2 × 32 × 8 × (4096/32) × 4096 × 1 × (16/8) / 1e9 = 2 × 32 × 8 × 128 × 4096 × 2 / 1e9 ≈ 0.8 GB`

### 🔹 **Overhead dynamique**
```
Overhead (GB) = base_framework + 0.015 × params_B
```
- `base_framework` : Overhead de base du framework (0.3 pour llama.cpp, 1.5 pour vLLM, etc.)
- `0.015 × params_B` : Scale linéaire avec la taille du modèle

**Exemple** : vLLM avec Llama 3 8B → `1.5 + 0.015 × 8 = 1.62 GB`

### 🔹 **Total RAM/VRAM**
```
Total (GB) = Poids + KV Cache + Overhead
```

### 🔹 **Entraînement**
```
Poids_entraînement (GB) = Poids × multiplicateur
```
- **Full Fine-Tuning** : ×8 (tous les poids + gradients + optimiseur)
- **LoRA** : ×5.5 (poids de base + adapteurs LoRA)
- **QLoRA** : ×3 (quantification des poids de base)

### 🔹 **Throughput estimé**
```
Tok/s ≈ (bande_passante_GB_s / total_model_GB) × 0.85
```
- `bande_passante_GB_s` : Bande passante mémoire du matériel (ex: 1008 GB/s pour RTX 4090)
- `0.85` : Efficacité mémoire estimée à 85%

---

## 🎯 Recommandations

### 🔹 **Pour l'inférence locale**
| Taille modèle | Quantification | Matériel recommandé | Débit estimé |
|---------------|----------------|---------------------|---------------|
| < 8B | INT4 | RTX 4060 Ti (16 GB) | 20-50 tok/s |
| 8B-24B | INT4 | RTX 4090 (24 GB) | 10-30 tok/s |
| 24B-70B | INT4 | A100 40 GB / H100 80 GB | 5-20 tok/s |
| > 70B | INT4 | H100 80 GB / B200 192 GB | 1-10 tok/s |

### 🔹 **Pour l'entraînement**
| Taille modèle | Méthode | Matériel minimum |
|---------------|---------|-------------------|
| 7B | Full FT | A100 80 GB (×8) |
| 7B | LoRA | RTX 4090 24 GB (×5.5) |
| 7B | QLoRA | RTX 4090 24 GB (×3) |
| 13B-70B | LoRA | A100 80 GB |
| > 70B | LoRA | H100 80 GB |

### 🔹 **Optimisations recommandées**
| Scénario | Optimisations KV Cache | Framework |
|----------|------------------------|-----------|
| Inférence locale | PagedAttention + FP8 KV | vLLM |
| Inférence légère | FlashAttention-2 | llama.cpp |
| Entraînement | PagedAttention + FP8 KV | vLLM / TGI |
| Apple Silicon | Aucune (MLX gère bien) | MLX |

---

## 🔍 Changelog

### **v5.2** (Actuelle)
- ✅ **Fix MoE débit** : Correction du calcul du débit pour les modèles MoE (seuls les poids actifs sont pris en compte)
- ✅ **models.js** : Base de données externalisée et structurée
- ✅ **Par famille** : Organisation des modèles par famille (Google, Alibaba, Mistral, etc.)
- ✅ **Ollama cmd** : Génération automatique des commandes `ollama run`
- ✅ **40+ modèles** : Ajout de Gemma 4, Qwen 3.6, Llama 3.3, DeepSeek V4, etc.

### **v5.1**
- ✅ **Comparateur de modèles** : Nouvelle section pour comparer plusieurs modèles
- ✅ **Recommandation matérielle** : Meilleur choix selon l'usage (Général, Math, Code, Conversation)
- ✅ **Efficacité énergétique** : Affichage du ratio tok/s/W

### **v4**
- ✅ **Overhead dynamique** : Calcul évolutif avec la taille du modèle
- ✅ **Optimisations KV Cache** : PagedAttention, FP8 KV, INT8 KV, FlashAttention
- ✅ **GQA-correct** : Formules corrigées pour les modèles GQA/MQA
- ✅ **Training réaliste** : Multiplicateurs ×3–×8 selon la méthode

### **v3**
- ✅ **Benchmarks** : Intégration HuggingFace Open LLM Leaderboard
- ✅ **Throughput estimé** : Calcul du débit en tok/s
- ✅ **Mode Entraînement** : Bascule Inférence/Entraînement

---

## 🤝 Contribuer

### 📥 **Ajouter un nouveau modèle**
1. Éditer `models.js`
2. Ajouter une entrée dans `PRESETS` avec les paramètres architecturaux
3. Ajouter une entrée dans `MODEL_FAMILIES` pour l'affichage dans le `<select>`
4. Optionnel : Ajouter un `hfid` pour la récupération des benchmarks

**Format d'un modèle dans `PRESETS`** :
```javascript
nom_unique: {
  params: 7,        // Nombre de milliards de paramètres
  hidden: 4096,    // Hidden size
  layers: 32,      // Nombre de couches
  heads: 32,       // Nombre de têtes d'attention
  kv: 8,          // Nombre de têtes KV (GQA si < heads)
  moe: 100,       // % de paramètres actifs (100 = dense)
  ctx: 4096,      // Contexte natif en tokens
  ollama: 'nom:tag', // Tag Ollama (optionnel)
  hfid: 'org/model', // ID HuggingFace (optionnel)
  cloud: false,   // true si cloud-only
  tags: ['tag1', 'tag2'], // Catégories pour le filtre
  note: 'Description' // Note affichée dans l'UI
}
```

### 🐛 **Signaler un bug**
1. Vérifier que le bug est reproductible
2. Noter les étapes pour le reproduire
3. Capturer une screenshot si nécessaire
4. Ouvrir une issue avec :
   - Version du calculateur
   - Navigateur utilisé
   - Étapes pour reproduire
   - Comportement attendu vs. observé

---

## 📜 Licence

Ce projet est **open source** et peut être utilisé librement pour des usages personnels ou professionnels.

- **Auteur** : Adrien (dredguer)
- **Version** : 5.2
- **Dernière mise à jour** : Juin 2025

---

## 🔗 Liens utiles

- [Ollama](https://ollama.com) – Déploiement local de LLM
- [HuggingFace Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) – Benchmarks des modèles
- [llm-stats.com](https://llm-stats.com) – Statistiques et comparatifs
- [vLLM](https://github.com/vllm-project/vllm) – Framework optimisé pour l'inférence
- [llama.cpp](https://github.com/ggerganov/llama.cpp) – Inférence LLM en C++

---

## 💡 Conseils

### 🔹 **Choisir la bonne quantification**
| Quantification | Taille relative | Précision | Usage recommandé |
|---------------|-----------------|-----------|-------------------|
| FP32 | 100% | ✅✅✅✅✅ | Entraînement |
| FP16 | 50% | ✅✅✅✅ | Inférence (GPU moderne) |
| INT8 | 25% | ✅✅✅ | Inférence (CPU/GPU) |
| INT4 | 12.5% | ✅✅ | Inférence locale |

### 🔹 **Optimiser le contexte**
- **Réduire le contexte** si la VRAM est limitée (ex: 2048 au lieu de 4096)
- **Utiliser GQA/MQA** : Les modèles avec `kv < heads` économisent de la VRAM
- **Batch size = 1** pour l'usage personnel (réduit le KV Cache)

### 🔹 **Choisir le bon framework**
- **llama.cpp / MLX** : Meilleur pour Apple Silicon et CPU
- **Ollama** : Simple et polyvalent
- **vLLM** : Optimisé pour le déploiement serveur (batch > 1)
- **PyTorch** : Flexible mais gourmand en mémoire

### 🔹 **Pour les modèles MoE**
- **Seuls les experts actifs** consomment de la VRAM pour le calcul
- **Tous les poids** sont chargés en RAM (même les experts inactifs)
- **Exemple** : DeepSeek V4 Flash (284B total, 13B actifs) → ~13B de VRAM pour le calcul, mais 284B de RAM pour les poids

---

## 📞 Support

Pour toute question ou suggestion, n'hésitez pas à :
- Ouvrir une issue sur le dépôt
- Contribuer via une Pull Request
- Partager vos retours d'expérience

---

> **⚡ Pro tip** : Utilise le **mode Auto** pour une configuration rapide, puis passe en **mode Manuel** pour affiner les paramètres selon tes besoins spécifiques !
