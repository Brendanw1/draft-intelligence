"use client";

import { useEffect, useState } from "react";
import { loadClass, loadDetail, loadIndex, loadManifest, loadMeta } from "./data";
import { ClassRow, DetailPlayer, IndexPlayer, Manifest, Meta } from "./types";

interface Loaded<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): Loaded<T> {
  const [state, setState] = useState<Loaded<T>>({ data: null, error: null, loading: true });
  useEffect(() => {
    let alive = true;
    setState((s) => ({ ...s, loading: true }));
    fn()
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((e) => alive && setState({ data: null, error: String(e), loading: false }));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

export function useIndex(): Loaded<IndexPlayer[]> {
  return useAsync(loadIndex, []);
}

export function useDetail(id: string | null): Loaded<DetailPlayer | null> {
  return useAsync(() => (id ? loadDetail(id) : Promise.resolve(null)), [id]);
}

export function useManifest(): Loaded<Manifest> {
  return useAsync(loadManifest, []);
}

export function useMeta(): Loaded<Meta> {
  return useAsync(loadMeta, []);
}

// cache-invalidator
export function useClassYear(year: number): Loaded<ClassRow[]> {
  return useAsync(() => loadClass(year), [year]);
}
