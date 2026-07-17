"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { loadMeta } from "@/lib/data";

const NAV = [
  { href: "/board", label: "Board" },
  { href: "/value", label: "Value Map" },
  { href: "/classes", label: "Classes" },
  { href: "/models", label: "Model Lab" },
  { href: "/models/audit", label: "Audit" },
  { href: "/methodology", label: "Methodology" },
];

export function TopBar() {
  const path = usePathname();
  const [vintage, setVintage] = useState<string | null>(null);
  const [dark, setDark] = useState(false);

  useEffect(() => {
    loadMeta()
      .then((m) => setVintage(new Date(m.generated_at).toISOString().slice(0, 10)))
      .catch(() => {});
    setDark(document.documentElement.dataset.theme === "dark");
  }, []);

  const toggleTheme = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.dataset.theme = next ? "dark" : "";
    try {
      localStorage.setItem("vtdi-theme", next ? "dark" : "light");
    } catch {}
  };

  return (
    <header className="sticky top-0 z-40 border-b border-rule bg-paper">
      <div className="flex h-12 items-center gap-6 px-4">
        <Link href="/board" className="flex items-baseline gap-2">
          <span
            className="text-[17px] font-semibold tracking-tight text-maroon"
            style={{ fontFamily: "var(--font-fraunces)" }}
          >
            VT Draft Intelligence
          </span>
          <span className="text-[11px] uppercase tracking-[0.14em] text-ink-3">
            2026
          </span>
        </Link>
        <nav className="flex items-center gap-1">
          {NAV.map((n) => {
            const active = path === n.href || path.startsWith(n.href + "/");
            return (
              <Link
                key={n.href}
                href={n.href}
                className={`rounded px-2.5 py-1 text-[13px] ${
                  active
                    ? "bg-maroon-soft font-semibold text-maroon"
                    : "text-ink-2 hover:text-ink"
                }`}
              >
                {n.label}
              </Link>
            );
          })}
        </nav>
        <div className="ml-auto flex items-center gap-3">
          {vintage && (
            <span
              className="text-[11px] text-ink-3"
              title="When the projection bundle was generated"
            >
              model vintage {vintage}
            </span>
          )}
          <button
            onClick={toggleTheme}
            className="rounded border border-rule px-2 py-0.5 text-[11px] text-ink-2 hover:border-rule-strong"
            aria-label="Toggle dark mode"
          >
            {dark ? "film room" : "daylight"}
          </button>
        </div>
      </div>
    </header>
  );
}
