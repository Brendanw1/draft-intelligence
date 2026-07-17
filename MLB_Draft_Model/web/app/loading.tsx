export default function Loading() {
  return (
    <div className="flex min-h-[60vh] items-center justify-center">
      <div className="flex flex-col items-center gap-3">
        <div className="h-3 w-48 animate-pulse rounded-sm bg-surface-1" />
        <div className="h-3 w-32 animate-pulse rounded-sm bg-surface-1" />
        <p className="mt-2 text-[12px] text-text-tertiary">Loading projections…</p>
      </div>
    </div>
  );
}
