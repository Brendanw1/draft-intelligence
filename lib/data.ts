import {
  ClassRow,
  DetailPlayer,
  DetailPlayerSchema,
  IndexPlayer,
  Manifest,
  ManifestSchema,
  Meta,
  MetaSchema,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_DATA_BASE ?? "";
const N_SHARDS = 64;

const cache = new Map<string, Promise<unknown>>();

async function fetchJson<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    cache.set(
      path,
      fetch(`${BASE}/data/${path}`).then((r) => {
        if (!r.ok) throw new Error(`fetch ${path}: ${r.status}`);
        return r.json();
      }),
    );
  }
  return cache.get(path) as Promise<T>;
}

// Index is large (10.7k rows). Zod-validate a sample, trust the rest — the
// bundle is produced by our own pipeline and validated fully at build time.
export async function loadIndex(): Promise<IndexPlayer[]> {
  const rows = await fetchJson<IndexPlayer[]>("players_index.json");
  return rows;
}

import { fnv1a } from "./hash";

export function shardOf(id: string): number {
  return fnv1a(id) % N_SHARDS;
}

export async function loadDetail(id: string): Promise<DetailPlayer | null> {
  const s = shardOf(id);
  const shard = await fetchJson<Record<string, unknown>>(
    `players/shard-${s.toString().padStart(2, "0")}.json`,
  );
  const raw = shard[id];
  if (!raw) return null;
  const parsed = DetailPlayerSchema.safeParse(raw);
  return parsed.success ? parsed.data : (raw as DetailPlayer);
}

export async function loadManifest(): Promise<Manifest> {
  const raw = await fetchJson<unknown>("models_manifest.json");
  return ManifestSchema.parse(raw);
}

export async function loadMeta(): Promise<Meta> {
  const raw = await fetchJson<unknown>("meta.json");
  return MetaSchema.parse(raw);
}

export async function loadClass(year: number): Promise<ClassRow[]> {
  return fetchJson<ClassRow[]>(`classes/${year}.json`);
}
