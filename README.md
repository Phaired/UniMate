# UniMate C++ / Vulkan

Portage C++20 de l'inférence de [UniMate](https://github.com/Friedrich-M/UniMate) (texte + GLB riggé →
animation squelettique par un transformeur de *flow matching*), sans Python ni PyTorch à l'exécution. Le
calcul tourne sur un backend Vulkan compute (GPU) ou sur un backend CPU de référence ; un viewer temps réel
Vulkan affiche et exporte le résultat.

> Ce dépôt est un travail dérivé d'**UniMate: One Unified Model to Animate Diverse Skeletons** (Linzhan Mou,
> Jiahui Lei, Zhiyang Dou, Chenyue Cai, Chaoyue Song, Adam Finkelstein, Szymon Rusinkiewicz — SIGGRAPH Asia
> 2026, [page du projet](https://linzhanmou.com/unimate/), [arXiv:2609.05415](https://arxiv.org/abs/2609.05415)).
> Le modèle, les poids, les données et le code PyTorch d'origine (entraînement, pipeline de données,
> inférence) sont ceux des auteurs et vivent dans le [dépôt upstream](https://github.com/Friedrich-M/UniMate) ;
> ce dépôt ne contient que le portage C++/Vulkan de l'inférence. La licence d'origine s'applique (`LICENSE`).
> L'historique git antérieur à ce portage est celui du code Python upstream.

Dans la suite, `reference/UniMate` désigne un clone du dépôt upstream placé à cet endroit (non versionné
ici) : il fournit les assets d'exemple (`assets/examples/`) et le code PyTorch dont les scripts de
conversion et de génération des goldens ont besoin.

```powershell
git clone --depth 1 https://github.com/Friedrich-M/UniMate reference/UniMate
```

**Périmètre par rapport à l'upstream** (`reference/UniMate/unimate/inference/sample.py`) :

- couvert : génération texte → mouvement pour un asset déjà converti (`rig_preprocess`), modèle publié
  `unimate_uniml3d_f60_v3` (60 images, 30 fps), encodeur `google/flan-t5-base`, CFG (échelle > 1, ou 1 =
  inconditionnel comme `sample.py`), solveurs `dopri5` (défaut upstream, adaptatif), `euler`, `heun`,
  récupération du mouvement et export GLB animé ;
- non couvert : entraînement, `rig_preprocess` lui-même (la préparation d'un nouveau rig reste en Python /
  Blender), inbetween / motion edit / motion expand, export BVH, fusion de bones (`merge`) faite par
  l'export Blender upstream (ici les joints absents du modèle gardent leur pose de repos) ;
- le bruit initial vient d'un générateur propre (xoshiro256**), pas de `torch.randn` : à seed égale, le
  mouvement diffère de l'upstream ; avec le *même* bruit (tests e2e), il est identique à ~1e-6 près.

## Prérequis

- Windows 10/11 x64, Visual Studio 2022 (MSVC, C++20), CMake ≥ 3.24 (testé avec 4.x), Git (les
  dépendances sont téléchargées par `FetchContent` à la première configuration : nlohmann_json,
  Vulkan-Headers, volk, VMA, glslang, GLFW, GLM, Dear ImGui, cgltf, stb).
- Un pilote Vulkan ≥ 1.2 pour le viewer et le backend GPU (aucun SDK Vulkan requis : volk charge le
  loader, les shaders sont compilés au build par glslang). Sans GPU, tout fonctionne sur CPU.
- Pour convertir les poids / régénérer les goldens seulement : Python 3 + `.venv` avec torch, numpy,
  transformers, torchdiffeq (les dépendances de `reference/UniMate`).

Environnement Python utilisé pour la conversion (CPU uniquement, Python 3.10) :

```powershell
uv venv --python 3.10 .venv
uv pip install --python .venv\Scripts\python.exe --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu" "numpy==2.2.6" "transformers==5.16.1" "sentencepiece==0.2.1" protobuf einops torchdiffeq "huggingface_hub==1.29.0" tyro rich loguru tqdm torch-geometric scipy regex
```

## Compilation

```powershell
cmake -S . -B build/all -G "Visual Studio 17 2022" -A x64
cmake --build build/all --config Release
```

Options (toutes `ON` par défaut) : `UM_BUILD_VK`, `UM_BUILD_T5`, `UM_BUILD_MODEL`, `UM_BUILD_PIPELINE`,
`UM_BUILD_APP`, `UM_BUILD_VIEWER`, `UM_BUILD_APPS`, `UM_BUILD_TESTS`. Avec `-DUM_BUILD_VK=OFF` les
applications sont CPU uniquement (`--backend auto` le signale). Exécutables produits :

- `build/all/apps/cli/Release/unimate_cli.exe`
- `build/all/apps/viewer/Release/unimate_viewer.exe` (et `unimate_viewer_demo.exe`, générateur factice)

## Poids et assets (conversion, une seule fois)

Les fichiers lus par le C++ sont au format UMTF (voir plus bas), dans `models/converted/`.

1. Télécharger les poids d'origine (par ex. avec `hf download`) :
   - `google/flan-t5-base` → `models/flan-t5-base/`
   - `Linzhan/UniMate`, dossier `unimate_uniml3d_f60_v3/` (config.json, dataset_stats.npy,
     checkpoints/) → `models/unimate/unimate_uniml3d_f60_v3/`
2. Convertir, dans cet ordre (indépendants entre eux, mais tous requis) :

```powershell
.venv\Scripts\python.exe tools/python/t5_export.py                 # -> models/converted/flan-t5-base.umtf
.venv\Scripts\python.exe tools/python/denoiser_export.py           # -> models/converted/unimate_uniml3d_f60_v3.umtf (poids EMA)
.venv\Scripts\python.exe tools/python/pipeline_convert_stats.py    # -> models/converted/unimate_uniml3d_f60_v3_stats.umtf
.venv\Scripts\python.exe tools/python/pipeline_convert_bundle.py --examples   # -> models/converted/bundles/{eagle,shark,unitree_go2}.umtf
```

3. Nouvel asset : le préparer avec `rig_preprocess` upstream (dossier contenant `cond.npy`,
   `summary.json`, `annotation.json` et le `.glb`), puis
   `.venv\Scripts\python.exe tools/python/pipeline_convert_bundle.py <dossier>` → `models/converted/bundles/<dossier>.umtf`.
   Donner un GLB sans bundle converti au CLI / viewer produit une erreur qui rappelle cette commande.

Goldens de test (`tests/golden/`, ~190 Mo, non versionnés : les tests qui en dépendent sont marqués
*Skipped* tant qu'ils sont absents). Pour les régénérer en exécutant le code upstream réel sur CPU :

```powershell
.venv\Scripts\python.exe tools/python/t5_golden.py
.venv\Scripts\python.exe tools/python/denoiser_golden.py
.venv\Scripts\python.exe tools/python/pipeline_golden.py --assets eagle shark
.venv\Scripts\python.exe tools/python/pipeline_golden.py --assets unitree_go2 --euler-only --skip-ode
```

Le dossier des modèles se change avec `--models DIR` ou la variable d'environnement `UNIMATE_MODELS`
(défaut : `<repo>/models/converted`).

## Utilisation

### CLI

```powershell
build\all\apps\cli\Release\unimate_cli.exe --asset eagle --prompt "An object flaps its wings." `
    --out eagle.glb --png eagle.png --png-frame rest,0,15,30,45
```

- `--asset` : nom de bundle (`eagle`, `shark`, `unitree_go2`), dossier d'asset, chemin `.glb` ou `.umtf` ;
  `--list-assets` liste les bundles convertis.
- `--seed N` (0), `--cfg S` (config du modèle : 3 ; 1 = inconditionnel), `--frames N` (≤ 60),
  `--solver dopri5|euler|heun` (dopri5), `--steps N` (grille torchdiffeq : N-1 pas, défaut 50 ;
  `--solver euler --steps 20` = 19 évaluations, rapide), `--backend auto|vulkan|cpu`.
- `--png F.png` rend des images hors écran avec le renderer du viewer (`--png-frame` liste ou `rest`,
  `--width/--height`, `--yaw/--pitch`, `--distance` pour reculer la caméra, `--no-skeleton`).
- `--noise fichier.umtf[:tenseur]` impose le bruit initial (comparaisons avec PyTorch).
- Affiche les temps : T5, conditionnement, échantillonnage (NFE, ms par appel de vitesse, pas acceptés /
  rejetés), récupération, total.

### Viewer

```powershell
build\all\apps\viewer\Release\unimate_viewer.exe eagle --backend auto
```

Panneau ImGui : prompt, seed, CFG, nombre d'images, solveur, *Generate*, liste des clips, lecture, export
GLB ; *Load* change d'asset (il faut un bundle converti). `--smoke-test SEC --capture out.png` génère
automatiquement puis ferme la fenêtre (test sans interaction). La génération tourne sur un thread de
travail pendant que le rendu utilise son propre device Vulkan sur le thread principal ; le backend de
calcul n'est utilisé que sous le mutex du `Generator` (le thread UI ne fait qu'enregistrer le chemin
de l'asset, chargé ensuite par le thread de travail).

## Architecture

```
src/core      um_core      Tensor / Backend abstraits, backend CPU de référence (AVX2), format UMTF
src/vk        um_vk        backend Vulkan compute (shaders GLSL -> SPIR-V embarqués, exécution différée)
src/t5        um_t5        tokenizer SentencePiece unigram + encodeur flan-t5-base (pooling moyen masqué)
src/model     um_model     débruiteur UniMate (graph attention + adaLN, RoPE spectral / temporel, CFG)
src/pipeline  um_pipeline  bundles, statistiques, dict de conditionnement, solveurs ODE (dopri5 fidèle à
                           torchdiffeq), récupération du mouvement, lecture / export GLB
src/app       um_app       um::Generator (tout ce qui précède derrière une classe), choix du backend,
                           résolution des assets / modèles
src/viewer    um_viewer    viewer Vulkan (skinning GPU, ImGui), rendu PNG hors écran
apps/         unimate_cli, unimate_viewer, unimate_viewer_demo
tests/e2e     um_e2e       parité bout en bout contre PyTorch
```

- **Backend** : `um::Backend` expose ~20 primitives fonctionnelles (unaire, binaire, matmul, linear,
  softmax, layer/rms norm, réductions, permute, slice, concat, gather, broadcast) et quelques composites
  surchargeables (attention, RoPE, modulate). T5 et le débruiteur ne connaissent que cette interface :
  le même code tourne sur CPU et sur GPU, le CPU servant de référence numérique. Un backend n'est pas
  thread-safe.
- **UMTF** : `"UMTF"`, version, en-tête JSON (`meta` libre + liste de tenseurs nom / dtype / shape /
  offset), puis données alignées sur 64 octets. Unique format d'échange Python ↔ C++ (poids, tokenizer,
  bundles, stats, goldens) ; côté Python : `tools/python/umtf.py`.
- **Generator** : `load_asset()` charge bundle + GLB + stats et encode une fois les noms de joints
  (cache par chaîne, le prompt aussi est mis en cache) ; `generate()` = `build_condition` →
  `Denoiser::set_condition` → bruit → `sample_flow` (vitesse CFG) → `recover_motion`. Sur Vulkan, le pool
  de buffers est libéré (`vulkan_trim`) après chaque génération.

## Validation (mesurée)

Tests : `ctest --test-dir build/all -C Release` (tests de modules + e2e). Les e2e longs sur CPU portent
le label `long` (`ctest -L e2e -LE long` pour les exclure), ceux du GPU le label `vulkan`.

E2E (`um_e2e`) : même légende, même bruit initial que l'exécution upstream réelle (`pipeline_golden.py`,
cfg 3), T5 + conditionnement + débruiteur + solveur + récupération en C++ ; écart max absolu sur les
joints valides de l'échantillon final (valeurs jusqu'à ~21) :

| asset / solveur | CPU max abs (RMS) | Vulkan max abs (RMS) | NFE C++ / torch | pas acceptés / rejetés |
|---|---|---|---|---|
| eagle Euler-20  | 2.4e-5 (1.2e-6) | 5.3e-5 (2.3e-6) | 19 / 19 | – |
| eagle dopri5    | 2.1e-5 (1.9e-6) | 3.3e-5 (2.3e-6) | 116 / 116 | 12/7 = 12/7 |
| shark Euler-20  | 6.7e-6 (6.0e-7) | 9.5e-6 (8.2e-7) | 19 / 19 | – |
| shark dopri5    | 4.1e-5 (4.0e-6) | 6.1e-5 (5.5e-6) | 128 / 128 | 13/8 = 13/8 |
| unitree_go2 Euler-20 | 1.1e-5 (8.2e-7) | 8.1e-6 (1.1e-6) | 19 / 19 | – |

Étapes intermédiaires : embeddings T5 ≤ 3.6e-7, dictionnaire de conditionnement ≤ 3.6e-7 (clés entières
identiques), premier appel de vitesse CFG ≤ 8.4e-5 (valeurs jusqu'à 20), mouvement récupéré (features,
trajectoire racine, positions FK, rotations BVH et rotations glTF exportées) ≤ 7e-6.
Pour dopri5, la séquence de pas acceptés / rejetés est identique à torchdiffeq ; les tailles de pas
dérivent jusqu'à ~8e-3 en relatif vers la fin (l'estimation d'erreur est dominée par le bruit d'arrondi
float32, que l'ordre des opérations change), sans effet visible sur l'échantillon final.

Temps (RTX 4070 Ti SUPER / CPU AVX2 du poste de test) : un appel de vitesse CFG (batch 2) ≈ 3.2 s sur CPU,
67–95 ms sur Vulkan (jusqu'à ~330 ms observé sous charge CPU concurrente) ; dopri5 complet (≈116–128 NFE)
≈ 8–12 s sur GPU contre ≈ 6–7 min sur CPU (PyTorch CPU upstream : ≈ 140 s) ; Euler-20 ≈ 1.3 s GPU,
≈ 60 s CPU. T5 : quelques ms par prompt sur GPU.

Contrôle visuel (`build/all/shots/`) : eagle (battements d'ailes), shark (nage vers l'avant), unitree_go2
(marche / trot vers l'avant) — maillage intact, squelette dans le maillage, poses différentes d'une image
à l'autre. Le viewer a généré sur Vulkan pendant son propre rendu Vulkan sans erreur.

## Limites connues

- Bruit initial non identique à `torch.randn` (voir plus haut) ; seed identique ≠ résultat upstream.
- Seuls les assets convertis par `pipeline_convert_bundle.py` sont utilisables ; pas de `rig_preprocess`
  en C++.
- La partie Blender de l'export upstream (import glTF, `compute_bone_keyframes`, export) est re-dérivée
  mathématiquement ; elle est vérifiée contre une ré-implémentation numpy, pas contre Blender.
- La caméra du rendu PNG est cadrée sur la pose de repos : un asset qui se déplace beaucoup (marche)
  sort du cadre, utiliser `--distance`.
- Le temps par appel de vitesse sur GPU varie (67–330 ms mesurés) selon la charge de la machine.
- Changement d'asset depuis l'UI du viewer pendant l'exécution : chemin de code en place (chargement
  paresseux sur le thread de travail) mais non testé automatiquement.
