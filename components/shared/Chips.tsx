"use client";

import { FLAG_HELP, FLAG_LABELS } from "@/lib/format";
import { GRADE_LABEL, gradeBg, gradeColor, typeBg, typeColor } from "@/lib/stats";
import { Grade, PlayerType } from "@/lib/types";

export function GradeChip({ grade, small }: { grade: Grade; small?: boolean }) {
  return (
    <span
      className={`inline-flex items-center rounded-sm font-semibold uppercase tracking-wide ${
        small ? "px-1.5 py-px text-[10px]" : "px-2 py-0.5 text-[11px]"
      }`}
      style={{ color: gradeColor(grade), background: gradeBg(grade) }}
    >
      {GRADE_LABEL[grade]}
    </span>
  );
}

export function TypeBadge({ type }: { type: PlayerType }) {
  return (
    <span
      className="inline-flex h-[18px] w-[18px] items-center justify-center rounded-sm text-[10px] font-bold"
      style={{ color: typeColor(type), background: typeBg(type) }}
      title={type === "hitter" ? "Hitter" : "Pitcher"}
    >
      {type === "hitter" ? "H" : "P"}
    </span>
  );
}

export function FlagChip({ flag }: { flag: string }) {
  return (
    <span
      className="inline-flex cursor-help items-center rounded-sm bg-flag-bg px-1.5 py-px text-[10px] font-medium text-flag"
      title={FLAG_HELP[flag] ?? flag}
    >
      ⚠ {FLAG_LABELS[flag] ?? flag}
    </span>
  );
}

export function FlagChips({
  flags,
  max = 3,
  nowrap,
}: {
  flags: string[];
  max?: number;
  nowrap?: boolean;
}) {
  if (!flags.length) return null;
  return (
    <span
      className={`inline-flex gap-1 ${nowrap ? "flex-nowrap overflow-hidden whitespace-nowrap" : "flex-wrap"}`}
    >
      {flags.slice(0, max).map((f) => (
        <FlagChip key={f} flag={f} />
      ))}
      {flags.length > max && (
        <span className="shrink-0 text-[10px] text-ink-3">+{flags.length - max}</span>
      )}
    </span>
  );
}

export function ConfDot({ level }: { level: "high" | "medium" | "low" | null }) {
  const label = level ?? "low";
  const filled = label === "high" ? 3 : label === "medium" ? 2 : 1;
  return (
    <span
      className="inline-flex items-center gap-[3px]"
      title={`Tier 1 confidence: ${label} (projected pick ${
        label === "high" ? "< 100" : label === "medium" ? "< 300" : "300+"
      })`}
      aria-label={`confidence ${label}`}
    >
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-[7px] w-[7px] rounded-full"
          style={{
            background: i < filled ? "var(--band-strong)" : "transparent",
            border: "1px solid var(--rule-strong)",
          }}
        />
      ))}
    </span>
  );
}
