"use client";

import { Suspense, useCallback, useMemo } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { BoardTable } from "@/components/board/BoardTable";
import { FilterRail } from "@/components/board/FilterRail";
import { StatTiles } from "@/components/chrome/StatTiles";
import {
  applyFilters,
  Filters,
  filtersToParams,
  paramsToFilters,
  sortRows,
} from "@/lib/filters";
import { useIndex } from "@/lib/hooks";

function BoardInner() {
  const { data: rows, loading, error } = useIndex();
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const filters = useMemo(() => paramsToFilters(new URLSearchParams(params)), [params]);

  const setFilters = useCallback(
    (f: Filters) => {
      const qs = filtersToParams(f).toString();
      router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
    },
    [router, pathname],
  );

  const filtered = useMemo(
    () => (rows ? applyFilters(rows, filters) : []),
    [rows, filters],
  );
  const sorted = useMemo(
    () => sortRows(filtered, filters.sort, filters.dir),
    [filtered, filters.sort, filters.dir],
  );

  const onSort = (key: string) => {
    if (filters.sort === key) {
      setFilters({ ...filters, dir: filters.dir === "desc" ? "asc" : "desc" });
    } else {
      setFilters({ ...filters, sort: key, dir: key === "pick" ? "asc" : "desc" });
    }
  };

  if (error)
    return (
      <div className="p-10 text-[13px] text-ink-2">
        Failed to load the projection bundle: {error}
      </div>
    );

  return (
    <div className="flex h-[calc(100vh-48px)]">
      <FilterRail rows={rows ?? []} filters={filters} setFilters={setFilters} />
      <div className="flex min-w-0 flex-1 flex-col">
        <StatTiles filtered={filtered} />
        {loading ? (
          <div className="p-10 text-[13px] text-ink-3">Loading 10,734 projections…</div>
        ) : (
          <BoardTable
            rows={sorted}
            typeMode={filters.type}
            grouped={filters.sort === "composite" && filters.dir === "desc"}
            sort={filters.sort}
            dir={filters.dir}
            onSort={onSort}
          />
        )}
      </div>
    </div>
  );
}

export default function BoardPage() {
  return (
    <Suspense>
      <BoardInner />
    </Suspense>
  );
}
