"use client";

import Link from "next/link";

const YEARS = [2021, 2022, 2023, 2024, 2025, 2026];

export default function ClassesPage() {
  return (
    <div className="mx-auto max-w-[900px] px-4 py-6">
      <h1
        className="text-[26px] font-semibold"
        style={{ fontFamily: "var(--font-fraunces)" }}
      >
        Draft Classes
      </h1>
      <p className="mt-1 max-w-[640px] text-[13px] leading-relaxed text-ink-2">
        How drafted college classes actually turned out — real names, real
        outcomes. This is how the tool teaches you its own error bars: pattern
        recognition on players whose stories already ended.
      </p>
      <div className="mt-5 grid grid-cols-2 gap-3 md:grid-cols-3">
        {YEARS.map((y) => (
          <Link
            key={y}
            href={`/classes/${y}/`}
            className="rounded border border-rule bg-paper-raised p-4 hover:border-rule-strong"
          >
            <div
              className="text-[22px] font-semibold"
              style={{ fontFamily: "var(--font-fraunces)" }}
            >
              {y}
            </div>
            <div className="text-[11px] text-ink-3">
              {y >= 2025
                ? "too recent for MLB outcomes"
                : "draft class + MiLB / MLB outcomes"}
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
