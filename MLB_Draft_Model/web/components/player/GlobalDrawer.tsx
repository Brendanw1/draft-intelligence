"use client";

import { useEffect } from "react";
import { useDetail } from "@/lib/hooks";
import { useUI } from "@/lib/store";
import { PlayerDossier } from "./PlayerDossier";

export function GlobalDrawer() {
  const { drawerId, closeDrawer } = useUI();
  const { data: player, loading } = useDetail(drawerId);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeDrawer();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [closeDrawer]);

  if (!drawerId) return null;

  return (
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true">
      <div
        className="absolute inset-0 bg-black/20"
        onClick={closeDrawer}
        aria-hidden
      />
      <div className="drawer-in absolute bottom-0 right-0 top-0 flex w-[540px] max-w-[92vw] flex-col border-l border-rule bg-paper shadow-[var(--shadow-drawer)]">
        <button
          onClick={closeDrawer}
          className="absolute right-3 top-3 z-10 rounded border border-rule bg-paper px-2 py-0.5 text-[12px] text-ink-2 hover:border-rule-strong"
          aria-label="Close player drawer"
        >
          esc ✕
        </button>
        {loading ? (
          <div className="p-8 text-[13px] text-ink-3">Loading dossier…</div>
        ) : player ? (
          <PlayerDossier p={player} inDrawer />
        ) : (
          <div className="p-8 text-[13px] text-ink-3">Player not found.</div>
        )}
      </div>
    </div>
  );
}
