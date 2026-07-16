"use client";

import Link from "next/link";
import { useUI } from "@/lib/store";

export function CompareTray() {
  const { compareIds, toggleCompare, clearCompare } = useUI();
  if (!compareIds.length) return null;
  return (
    <div className="fixed bottom-0 left-1/2 z-40 -translate-x-1/2 pb-3">
      <div className="flex items-center gap-2 rounded border border-rule-strong bg-paper-raised px-3 py-2 shadow-[var(--shadow-pop)]">
        <span className="text-[11px] font-semibold uppercase tracking-wide text-ink-3">
          Compare
        </span>
        {compareIds.map((id) => (
          <button
            key={id}
            onClick={() => toggleCompare(id)}
            className="rounded-sm bg-paper-sunken px-2 py-0.5 text-[12px] text-ink-2 hover:text-ink"
            title="Remove"
          >
            {id.split("-").slice(0, -2).join(" ")} ✕
          </button>
        ))}
        <Link
          href={`/compare/?ids=${compareIds.map(encodeURIComponent).join(",")}`}
          className={`rounded px-3 py-1 text-[12px] font-semibold ${
            compareIds.length >= 2
              ? "bg-maroon text-white"
              : "pointer-events-none bg-paper-sunken text-ink-3"
          }`}
        >
          Open side-by-side
        </Link>
        <button
          onClick={clearCompare}
          className="text-[11px] text-ink-3 hover:text-ink"
        >
          clear
        </button>
      </div>
    </div>
  );
}
