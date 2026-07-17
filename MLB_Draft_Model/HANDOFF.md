# Handoff: Three-Tier Model Pipeline — State of the Project

**Date:** 2026-07-16  
**Repository:** `https://github.com/Brendanw1/vt-draft-intelligence`  
**Branch:** `main`  
**HEAD:** `150bddd` (Update README: three-tier model documentation)  
**Production URL:** `https://web-beige-eta-46.vercel.app` (auto-aliased)  

---

## 1. What Was Built — Three-Tier Model Architecture

We replaced the single-model (Tier 2 only) pipeline with a three-tier system:

### Tier 1: Projected Draft Pick (XGBoost Regressor)

| | Hitters | Pitchers |
|---|---|---|
| **Algorithm** | XGBoost regressor | XGBoost regressor |
| **Training size** | 1,189 | 1,608 |
| **CV MAE** | ~82 picks | ~91 picks |
| **Top features** | Age, strength_x_OPS, **conf_strength** | Age, **conf_strength**, K-BB% |

**Key change:** `conf_strength` (continuous, empirical draft rate ratio) replaced the broken categorical `conference_tier` (4 tiers). This fixed elite ACC/SEC players being undervalued.

**Example corrections:**
- Drew Burress: pick 226 → **16** (correctly recognized as 1st-round talent)
- Vahn Lackey: pick 196 → **38** (correctly recognized as 2nd-round)

**Artifacts:** `models/artifacts_full/`
- `{fg_draft,tier1_features}_{hitter,pitcher}.json` — model metadata, features, importance

---

### Tier 2: Draft Probability (XGBoost Classifier + Platt Calibration)

| | Hitters | Pitchers |
|---|---|---|
| **Algorithm** | XGBoost classifier + Platt scaling | XGBoost classifier + Platt scaling |
| **Training size** | 31,988 player-seasons | 31,196 player-seasons |
| **AUC** | ~0.988 | ~0.987 |
| **Top feature** | height_inches | height_inches |

**Key changes:**
- `conf_strength` replaced categorical conference tiers
- Conference-adjusted stats (`wOBA_adj`, `ERA_adj`, `OPS_adj`, etc.) replace raw stats
- Interaction features (`strength_x_{stat}`) explicitly model conference penalty
- Raw `mlb_probability` (conference-adjusted) is the primary signal, not Platt-calibrated

**Artifacts:** `models/artifacts_full/`
- `tier2_full_{hitter,pitcher}.json` — model files
- `tier2_full_features_{hitter,pitcher}.json` — metadata
- `calibration_curve_*.json`, `calibrator_*_{isotonic,platt}*.pkl` — calibrators

---

### Tier 3: MLB Arrival Probability (NEW — Elastic Net + Nearest Neighbor)

This is new. Predicts P(MLB debut | drafted) — the probability that a player who gets drafted will eventually reach MLB.

| | Hitters | Pitchers |
|---|---|---|
| **Algorithm** | Elastic Net (prior-offset) | Elastic Net (prior-offset) |
| **Training size** | 549 drafted players (94 debut) | 713 drafted players (132 debut) |
| **CV AUC** | 0.786 ± 0.043 | 0.788 ± 0.045 |
| **Features** | 10 (Age, conf_strength, wOBA_adj, OPS_adj, BB%, K%, height, bmi, round_logit_prior, nn_mlb_rate) | 10 (Age, conf_strength, ERA_adj, FIP_adj, K/9_adj, BB/9_adj, height, bmi, round_logit_prior, nn_mlb_rate) |
| **Top coefficient** | round_logit_prior (+1.03) | round_logit_prior (+1.14) |

**Key design decisions:**
- **Prior-offset architecture:** Uses `round_logit_prior` — the logit of each round's empirical MLB debut rate (1st round = 61%, 5th = 27%, 10th = 12%, 20th = 3%). This anchors predictions to the correct historical baseline for each round.
- **`nn_mlb_rate`:** For each 2026 prospect, finds 20 nearest drafted players by Euclidean distance on features. The % of those 20 who reached MLB is a feature in the Elastic Net. This captures "profile similarity to MLB arrivals" that raw stats miss.
- **Adjusted stats throughout:** All stat inputs are conference-adjusted (`wOBA_adj`, `ERA_adj`, etc.).

**Artifacts:** `models/artifacts_full/`
- `tier3_mlb_{hitter,pitcher}.pkl` — trained Elastic Net + feature list
- `tier3_nn_{hitter,pitcher}.pkl` — nearest-neighbor reference pools (drafted player feature vectors + MLB outcomes)
- `tier3_features_{hitter,pitcher}.json` — metadata, coefficients, CV scores

---

### Conference Strength System

**Script:** `scripts/compute_conference_strength.py`
**Artifacts:** `models/artifacts_full/conference_strength.json`, `models/artifacts_full/conference_stats.json`

`conf_strength` = empirical draft rate ratio (conference draft rate / global draft rate).

Key values:

| Conference | conf_strength | Interpretation |
|---|---|---|
| DI Independent | 3.29 | 3.29× more draft picks than average |
| SEC | 2.98 | Strongest major conference |
| ACC | 2.85 | Elite |
| Big Ten | 1.56 | Above average |
| Big West | 1.17 | Slightly above |
| SWAC | 0.09 | Very underrepresented |
| MEAC | 0.17 | Very underrepresented |

---

## 2. Data Pipeline — Scripts in Order

The production pipeline is in `scripts/`. Each script is independently runnable. The inference pipeline (`infer_2026.py`) is the main integration point.

| Script | What It Does | Runs In |
|---|---|---|
| `compute_conference_strength.py` | Computes conf_strength from historical draft rates | Training |
| `train_tier3_mlb_arrival.py` | Trains Tier 3 Elastic Net + NN reference pool | Training (one-time) |
| `train_tier2_full.py` | Retrains Tier 2 with conf_strength + adj stats | Training |
| `train_tier1_model.py` | Retrains Tier 1 projected pick model | Training |
| `infer_2026.py` | **Main inference script** — runs T1, T2, T3 for all 10,734 2026 prospects | Daily/weekly |
| `export_frontend_data.py` | Generates `web/public/data/` (players_index.json, shards, manifest, meta, classes) | After inference |
| `sync_data_to_r2.py` | Uploads frontend data to Cloudflare R2 | After export |

---

## 3. Frontend Data — What's In The Bundle

Data lives in `web/public/data/` (deployed with app) and is also synced to R2 for CDN serving.

**Vercel env var:** `NEXT_PUBLIC_DATA_BASE=https://pub-a7408699ec9340f8bd4be40dcf9e45de.r2.dev`

### Files

| File | Size | Contains |
|---|---|---|
| `players_index.json` | 9.2 MB | All 10,734 players with index-level fields |
| `meta.json` | 344 B | Generation timestamp, counts, backtest stats |
| `models_manifest.json` | 13.6 KB | Model cards for Model Lab |
| `players/shard-00..63.json` | ~25 MB total | Per-player detail records (split 64 ways) |
| `classes/2021..2026.json` | ~600 KB each | Historical draft class records |

### Index Fields (every player in `players_index.json`)

```typescript
{
  id: string,                    // "tomas-valincius-msst-p"
  name: string,                  // "Tomas Valincius"
  type: "hitter" | "pitcher",
  school: string,                // "Mississippi State"
  school_abb: string,            // "MSST"
  conference: string | null,     // "SEC"
  age: number | null,
  proj_pick: number | null,      // Tier 1: projected pick
  proj_round: number | null,     // Tier 1: projected round
  pick_band: [number, number] | null, // Tier 1: confidence band
  t1_confidence: "high"|"medium"|"low" | null,
  mlb_p: number | null,          // Tier 2: calibrated MLB probability
  mlb_p_raw: number | null,      // Tier 2: raw (conference-adjusted) score
  mlb_p_iso: number | null,      // Tier 2: isotonic calibration
  mlb_arrival: number | null,    // ** Tier 3: P(MLB debut | drafted) **
  nn_mlb_rate: number | null,    // ** Tier 3: nearest-neighbor MLB rate **
  hist_rate: number | null,      // Tier 2: calibration bin historical rate
  composite: number | null,      // Composite score (40% slot + 60% MLB%)
  grade: "elite"|"high"|"medium"|"low",
  sample: { pa: number|null, ip: number|null },
  flags: string[],               // e.g. "low_pa", "low_ip", "wide_spread"
  key_stats: Record<string, number|null>,
  height_inches: number | null,
  bmi: number | null,
  draftability_score: number | null,
  conference_tier: number | null,
  pctl: Record<string, number>,  // Percentile ranks
}
```

**NEW fields (added by Tier 3):**
- `mlb_arrival` — Elastic Net probability 0.0–42.6% range. Mean ~8%. Max is Dylan Volantis at 42.6%.
- `nn_mlb_rate` — Comp-based rate 0.0–4.76% range. Capped by the small reference pool (5% max across 20 comps).

### Detail Fields (in shard files, per player)

All index fields plus:
```typescript
{
  xMLBAMID: number | null,
  seasons: Record<string, number|null>[],  // Multi-year stat lines
  pctl: Record<string, number>,            // Percentile ranks within position
  comps: { name, school, year, pick, round, reached_mlb, peak_level, dist }[], // Stat-profile comps
  height_display: string | null,
}
```

---

## 4. What the Frontend Needs to Do

The data is live and served correctly from R2. The frontend just needs to **display it**.

### 4.1 — Schema: Add Fields to types.ts

**File:** `web/lib/types.ts`

`IndexPlayerSchema` needs two new fields added (they exist in the data but Zod strips unknown fields by default — actually `loadIndex` doesn't validate with Zod, but `DetailPlayerSchema` extends `IndexPlayerSchema` and gets parsed in `loadDetail`, so DetailPlayerSchema needs them):

```typescript
// Add to IndexPlayerSchema:
  mlb_arrival: z.number().nullable(),
  nn_mlb_rate: z.number().nullable(),
```

Note: `loadIndex()` currently does NOT validate with Zod (it returns raw JSON), so the board already receives these fields. But adding them to the schema enables type-safe access. `loadDetail()` DOES validate with `DetailPlayerSchema.parse()`, and since the shard data has these fields but the schema doesn't, Zod strips them silently. Adding them to the schema fixes this.

### 4.2 — Board Table: Add "MLB Arrival" Column

**File:** `web/components/board/BoardTable.tsx`

Add a sortable column between `MLB%` (line 123) and `Conf` (line 124):

```tsx
<div className="w-[100px] shrink-0">
  <Th label="Arrival" sortKey="mlb_arrival" title="Tier 3: Probability of reaching MLB if drafted. Round-anchored prior + Elastic Net + NN comp rate." />
</div>
```

In the row render (after line 219), show the arrival probability as a compact bar:
- Main bar: `mlb_arrival` value (e.g., "42.6%")
- Small indicator: `nn_mlb_rate` as a reference dot (e.g., "NN 4.8%")
- Color by tier: high (>20%) green, medium (5–20%) amber, low (<5%) gray

Render logic (insert after the `ConfDot` div, around line 221):

```tsx
<div className="w-[100px] shrink-0 px-2">
  <ProbCell p={p.mlb_arrival} raw={p.nn_mlb_rate} hist={null} width={48} />
</div>
```

The existing `ProbCell` component in `web/components/shared/ProbCell.tsx` can be reused — it renders a compact bar with a reference tick. Pass `mlb_arrival` as `p` and `nn_mlb_rate` as `raw`.

### 4.3 — Player Dossier: Add "MLB Outlook" Section

**File:** `web/components/player/PlayerDossier.tsx`

Insert after the `VerdictRow` (line 236) and before "Physical profile" (line 239):

A new section showing:
- Tier 3 arrival probability (large number)
- NN comp rate (reference)
- Round-specific historical baseline (computed per prospect from their projected round)
- Interpretation text: "X× the historical round-Y baseline"

Something like:

```tsx
{/* MLB Outlook — Tier 3 */}
<div className="border-t border-rule px-4 py-3">
  <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.13em] text-ink-3">
    MLB Outlook (if drafted)
  </div>
  <div className="grid grid-cols-3 gap-3">
    <div>
      <div className="text-[10px] text-ink-3">Arrival probability</div>
      <div className="text-[18px] font-semibold leading-tight">
        {fmtPct(p.mlb_arrival)}
      </div>
    </div>
    <div>
      <div className="text-[10px] text-ink-3">NN comp rate</div>
      <div className="text-[18px] font-semibold leading-tight">
        {fmtPct(p.nn_mlb_rate)}
      </div>
    </div>
  </div>
</div>
```

The detail shard also has `comps` (20 nearest stat-profile comps from the Tier 2 similarity engine). These are different from the Tier 3 NN comps — the Tier 3 NN comps are built into `nn_mlb_rate` as an aggregate, not exposed individually. If you want to show individual Tier 3 comps, that would require modifying the inference pipeline to store them.

### 4.4 — Model Lab: Add Tier 3 Model Cards

**File:** The Tier 3 models are not in `models_manifest.json`. The export script (`export_frontend_data.py`) needs to be updated to include them.

The manifest is generated in `export_frontend_data.py` starting around line 600. A new block needs to be added for Tier 3 models, reading from `models/artifacts_full/tier3_features_{hitter,pitcher}.json` and creating entries like:

```json
"tier3-mlb-arrival-hitter": {
  "artifact": "tier3_mlb_hitter.pkl",
  "tier": 3,
  "type": "hitter",
  "display_name": "MLB Arrival Model — Hitters",
  "target": "P(MLB debut | drafted)",
  "algorithm": "Elastic Net (prior-offset logit)",
  "training_population": "D1 college draftees 2021-2023",
  "n_train": 549,
  "n_positive": 94,
  "base_rate": 0.171,
  "n_features": 10,
  "features": ["Age","conf_strength","wOBA_adj","..."],
  "importance": [...coefficients...],
  "notes": "Prior-offset architecture: round_logit_prior anchors predictions to empirical round-specific MLB debut rates."
}
```

### 4.5 — Compare Page: Add Tier 3 Fields

**File:** `web/app/compare/page.tsx`

Add `mlb_arrival` and `nn_mlb_rate` to the comparison columns shown for each player.

---

## 5. R2 / Deployment Setup

### R2 Bucket

- **Endpoint:** `https://af66e2cdfa211f99f117eb8101060ab7.r2.cloudflarestorage.com`
- **Bucket:** `mlbdraftcol`
- **Access Key:** `71b99729988b9edeee9bbf4d64d8f516`
- **Secret Key:** `ec089c9c9c5963a52c2d591aa1801626b31074dbf284960a1db57ad41dfbe52c`
- **Public URL:** `https://pub-a7408699ec9340f8bd4be40dcf9e45de.r2.dev`
- **rclone remote name:** `r2` (configured in `~/.config/rclone/rclone.conf`)
- **rclone path:** `r2:mlbdraftcol/data/`

### Sync Command

```bash
rclone copy web/public/data/ r2:mlbdraftcol/data/ \
  --include "players_index.json" \
  --include "meta.json" \
  --include "models_manifest.json" \
  --include "classes/*.json" \
  --include "players/shard-*.json" \
  --progress
```

### Vercel Environment Variables

| Variable | Value | Scope |
|---|---|---|
| `NEXT_PUBLIC_DATA_BASE` | `https://pub-a7408699ec9340f8bd4be40dcf9e45de.r2.dev` | Production |

When this is set, the frontend fetches data from R2 instead of the bundled `web/public/data/`. The source of truth is the local `web/public/data/` folder — R2 is just a CDN copy.

### Deployment

```bash
cd web && vercel --prod
```

---

## 6. Outdated Content on the Site

### Methodology Page

**File:** `web/app/methodology/page.tsx` (static content in the component)

The following statements are now WRONG and need updating:

1. **"No conference adjustment"** — This was fixed. Conference-adjusted stats (`wOBA_adj`, `ERA_adj`) and `conf_strength` are now first-class features in all tiers. The text should say something like "Conference-adjusted: all stats are normalized by conference strength to avoid overrating small-conference production."

2. **No mention of Tier 3** — The pipeline description only describes two models. Needs a paragraph on the Tier 3 MLB arrival model with prior-offset architecture.

3. **"Platt calibration"** — The calibration description should mention that raw `mlb_probability` (uncalibrated, conference-adjusted) is the primary signal on the board, with calibration as secondary.

### Known Limitations Section

The "No conference adjustment" bullet should be replaced with something like:
"Conference adjustment is based on empirical draft rates (2021-2025). This is a reasonable proxy for competitive strength but not a direct measure of conference quality. A few small conferences (DI Independent) have high draft rates due to individual programs, not overall strength."

Add a bullet for Tier 3:
"Tier 3 (MLB arrival) is trained on 2021-2023 draftees only — the most recent classes haven't had time to debut. Rates will shift as 2024-2025 draftees reach MLB."

---

## 7. Known Issues & Caveats

1. **nn_mlb_rate is low-capped** — The NN reference pool is 1,112 (hitter) / 1,488 (pitcher) drafted players. With 20 comps per prospect and a ~12% debut rate, the max `nn_mlb_rate` is ~5%. This is still a useful signal in the Elastic Net, but looks low when displayed standalone. Don't expect comp rates above 5%.

2. **Tier 2 artifact path mismatch** — The old `models/artifacts/tier2_features_{hitter,pitcher}.json` are pre-conference-adjustment. The current files are in `models/artifacts_full/tier2_full_features_*.json`. The `export_frontend_data.py` script reads from both — check that the manifest doesn't reference stale artifacts.

3. **Tier 3 not in manifest** — The Model Lab page will not show Tier 3 models until `export_frontend_data.py` is updated to add them.

4. **No custom domain on Vercel** — The site deploys to auto-generated vercel.app URLs. No custom domain (like `mlbdraft.vtbaseball.dev`) is configured.

5. **Methodology page is static JSX** — Content is hard-coded in `web/app/methodology/page.tsx`. There's no CMS or markdown source. Edits must be made to the React component directly.

---

## 8. Example Queries to Verify Data

```python
# Connect to R2 and check data
import requests, json
r = requests.get("https://pub-a7408699ec9340f8bd4be40dcf9e45de.r2.dev/data/players_index.json")
data = r.json()
print(f"{len(data)} players")
top = sorted(data, key=lambda x: x.get('mlb_arrival',0), reverse=True)[0]
print(f"Top arrival: {top['name']} = {top['mlb_arrival']:.1%}")

# Check specific player
burress = [p for p in data if 'Burress' in p['name']][0]
print(f"Drew Burress: proj_pick={burress['proj_pick']}, mlb_p={burress['mlb_p']:.1%}, arrival={burress['mlb_arrival']:.1%}")
```

Should output:
- 10734 players
- Top arrival: Dylan Volantis = 42.6%
- Drew Burress: proj_pick=16, mlb_p=95.8%, arrival=10.7%

---

## 9. Quick Reference — File Paths

| Purpose | Path |
|---|---|
| Training DB | `data/training/training_set.csv` |
| Draft data | `data/draft/draft_college_picks.json` |
| Tier 1 model | `models/artifacts_full/fg_draft_{hitter,pitcher}.json` + `tier1_features_*.json` |
| Tier 2 model | `models/artifacts_full/tier2_full_{hitter,pitcher}.json` + `tier2_full_features_*.json` |
| Tier 3 model | `models/artifacts_full/tier3_mlb_{hitter,pitcher}.pkl` + `tier3_features_*.json` |
| NN ref pool | `models/artifacts_full/tier3_nn_{hitter,pitcher}.pkl` |
| Conference data | `models/artifacts_full/conference_strength.json`, `conference_stats.json` |
| Inference pipeline | `scripts/infer_2026.py` |
| Frontend export | `scripts/export_frontend_data.py` |
| Frontend data | `web/public/data/` |
| Type definitions | `web/lib/types.ts` |
| Board table | `web/components/board/BoardTable.tsx` |
| Player dossier | `web/components/player/PlayerDossier.tsx` |
| Methodology page | `web/app/methodology/page.tsx` |
| Compare page | `web/app/compare/page.tsx` |
| R2 sync | `scripts/sync_data_to_r2.py` |
| R2 config | `~/.config/rclone/rclone.conf` |
| Vercel project | `.vercel/project.json` |
