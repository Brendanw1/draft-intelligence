// FNV-1a 32-bit — mirrored in scripts/export_frontend_data.py; keep in sync.
export function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  const bytes = new TextEncoder().encode(s);
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i];
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}
