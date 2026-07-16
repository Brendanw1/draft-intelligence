import { z } from "zod";

export const GradeSchema = z.enum(["elite", "high", "medium", "low"]);
export type Grade = z.infer<typeof GradeSchema>;

export const PlayerTypeSchema = z.enum(["hitter", "pitcher"]);
export type PlayerType = z.infer<typeof PlayerTypeSchema>;

export const IndexPlayerSchema = z.object({
  id: z.string(),
  name: z.string(),
  type: PlayerTypeSchema,
  school: z.string(),
  school_abb: z.string(),
  conference: z.string().nullable(),
  age: z.number().nullable(),
  proj_pick: z.number().nullable(),
  proj_round: z.number().nullable(),
  pick_band: z.tuple([z.number(), z.number()]).nullable(),
  t1_confidence: z.enum(["high", "medium", "low"]).nullable(),
  mlb_p: z.number().nullable(),
  mlb_p_raw: z.number().nullable(),
  mlb_p_iso: z.number().nullable(),
  hist_rate: z.number().nullable(),
  composite: z.number().nullable(),
  grade: GradeSchema,
  sample: z.object({ pa: z.number().nullable(), ip: z.number().nullable() }),
  flags: z.array(z.string()),
  key_stats: z.record(z.string(), z.number().nullable()),
});
export type IndexPlayer = z.infer<typeof IndexPlayerSchema>;

export const CompSchema = z.object({
  name: z.string().nullable(),
  school: z.string().nullable(),
  year: z.number().nullable(),
  pick: z.number().nullable(),
  round: z.number().nullable(),
  reached_mlb: z.boolean(),
  peak_level: z.string().nullable(),
  dist: z.number().nullable(),
});
export type Comp = z.infer<typeof CompSchema>;

export const DetailPlayerSchema = IndexPlayerSchema.extend({
  xMLBAMID: z.number().nullable(),
  seasons: z.array(z.record(z.string(), z.number().nullable())),
  pctl: z.record(z.string(), z.number()),
  comps: z.array(CompSchema),
});
export type DetailPlayer = z.infer<typeof DetailPlayerSchema>;

export const BacktestRowSchema = z.object({
  test_year: z.number(),
  type: PlayerTypeSchema,
  n_train: z.number(),
  n_test: z.number(),
  features: z.number(),
  mae: z.number(),
  baseline_mae: z.number(),
  r2: z.number(),
  baseline_r2: z.number(),
  spearman_rho: z.number(),
  top100_overlap: z.number(),
});
export type BacktestRow = z.infer<typeof BacktestRowSchema>;

export const CalibrationBinSchema = z.object({
  bin: z.string(),
  count: z.number(),
  pred_mean: z.number(),
  actual_rate: z.number(),
});
export type CalibrationBin = z.infer<typeof CalibrationBinSchema>;

export const ModelCardSchema = z.object({
  artifact: z.string(),
  tier: z.number(),
  type: PlayerTypeSchema,
  display_name: z.string(),
  target: z.string(),
  algorithm: z.string(),
  training_population: z.string(),
  n_train: z.number(),
  n_positive: z.number().optional(),
  base_rate: z.number().optional(),
  n_features: z.number(),
  features: z.array(z.string()),
  importance: z.array(z.object({ feature: z.string(), importance: z.number() })),
  backtest: z.array(BacktestRowSchema).optional(),
  flagged_features: z.array(z.string()).optional(),
  calibration: z
    .object({
      type: z.string(),
      n: z.number(),
      ece: z.number(),
      mean_pred: z.number(),
      mean_actual: z.number(),
      bias: z.number(),
      bins: z.array(CalibrationBinSchema),
    })
    .nullable()
    .optional(),
  recalibration: z
    .object({
      type: z.string(),
      n_train: z.number(),
      n_val: z.number(),
      mlb_rate_train: z.number(),
      mlb_rate_val: z.number(),
      brier_raw: z.number(),
      brier_platt: z.number(),
      brier_iso: z.number(),
      mean_raw_val: z.number(),
      mean_platt_val: z.number(),
      mean_iso_val: z.number(),
      mean_actual_val: z.number(),
    })
    .nullable()
    .optional(),
  notes: z.string().optional(),
});
export type ModelCard = z.infer<typeof ModelCardSchema>;
export const ManifestSchema = z.record(z.string(), ModelCardSchema);
export type Manifest = z.infer<typeof ManifestSchema>;

export const ClassRowSchema = z.object({
  name: z.string().nullable(),
  type: PlayerTypeSchema,
  school: z.string().nullable(),
  team: z.string().nullable(),
  position: z.string().nullable(),
  round: z.number().nullable(),
  pick: z.number().nullable(),
  bonus: z.number().nullable(),
  age: z.number().nullable(),
  reached_mlb: z.boolean().nullable(),
  peak_level: z.string().nullable(),
  first_milb_ops: z.number().nullable(),
  stats: z.record(z.string(), z.number().nullable()),
});
export type ClassRow = z.infer<typeof ClassRowSchema>;

export const MetaSchema = z.object({
  generated_at: z.string(),
  season: z.number(),
  players: z.number(),
  hitters: z.number(),
  pitchers: z.number(),
  grades: z.record(z.string(), z.number()),
  pick_band_mae: z.record(z.string(), z.number()),
  backtest_year: z.number(),
  min_sample: z.object({ pa: z.number(), ip: z.number() }),
  class_years: z.array(z.number()),
  conference_coverage: z.number(),
});
export type Meta = z.infer<typeof MetaSchema>;
