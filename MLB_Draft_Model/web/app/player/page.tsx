"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { useDetail } from "@/lib/hooks";
import { PlayerDossier } from "@/components/player/PlayerDossier";

function PlayerInner() {
  const params = useSearchParams();
  const id = params.get("id");
  const { data: player, loading } = useDetail(id);

  if (!id)
    return <div className="p-10 text-[13px] text-ink-3">No player specified.</div>;
  if (loading)
    return <div className="p-10 text-[13px] text-ink-3">Loading dossier…</div>;
  if (!player)
    return <div className="p-10 text-[13px] text-ink-3">Player not found: {id}</div>;

  return (
    <div className="mx-auto max-w-[860px] border-x border-rule bg-paper min-h-[calc(100vh-48px)]">
      <PlayerDossier p={player} />
    </div>
  );
}

export default function PlayerPage() {
  return (
    <Suspense>
      <PlayerInner />
    </Suspense>
  );
}
