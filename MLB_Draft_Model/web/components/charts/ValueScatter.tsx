"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { IndexPlayer } from "@/lib/types";
import { fmtPct } from "@/lib/format";
import { useUI } from "@/lib/store";
import { GradeChip, TypeBadge } from "@/components/shared/Chips";

const M = { top: 24, right: 24, bottom: 44, left: 52 };
const PICK_MAX = 620;

/** css var → concrete color for canvas */
function cssVar(name: string): string {
  if (typeof window === "undefined") return "#888";
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

export function ValueScatter({
  rows,
  showPitcherCaution,
  onBrush,
}: {
  rows: IndexPlayer[];
  showPitcherCaution: boolean;
  onBrush: (sel: { pickMin: number; pickMax: number; probMin: number; probMax: number } | null) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 900, h: 560 });
  const [hover, setHover] = useState<IndexPlayer | null>(null);
  const [hoverXY, setHoverXY] = useState<[number, number]>([0, 0]);
  const [drag, setDrag] = useState<null | { x0: number; y0: number; x1: number; y1: number }>(null);
  const { openDrawer } = useUI();

  const pts = useMemo(
    () => rows.filter((r) => r.proj_pick != null && r.mlb_p != null),
    [rows],
  );

  // calibrated probabilities compress low — scale the axis to the data
  const yMax = useMemo(() => {
    const m = Math.max(0.1, ...pts.map((p) => p.mlb_p!));
    return Math.min(1, Math.ceil((m * 1.15) / 0.05) * 0.05);
  }, [pts]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      setSize({ w: el.clientWidth, h: Math.max(420, Math.min(640, window.innerHeight - 240)) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const xScale = useCallback(
    (pick: number) =>
      M.left + (Math.sqrt(pick) / Math.sqrt(PICK_MAX)) * (size.w - M.left - M.right),
    [size.w],
  );
  const yScale = useCallback(
    (p: number) => M.top + (1 - p / yMax) * (size.h - M.top - M.bottom),
    [size.h, yMax],
  );
  const xInv = useCallback(
    (px: number) => {
      const f = (px - M.left) / (size.w - M.left - M.right);
      return Math.pow(Math.max(0, Math.min(1, f)) * Math.sqrt(PICK_MAX), 2);
    },
    [size.w],
  );
  const yInv = useCallback(
    (py: number) =>
      (1 - Math.max(0, Math.min(1, (py - M.top) / (size.h - M.top - M.bottom)))) * yMax,
    [size.h, yMax],
  );

  // draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.w * dpr;
    canvas.height = size.h * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, size.w, size.h);

    const colors = {
      elite: cssVar("--grade-elite"),
      high: cssVar("--grade-high"),
      medium: cssVar("--grade-medium"),
      low: cssVar("--grade-low"),
      rule: cssVar("--rule"),
      ruleStrong: cssVar("--rule-strong"),
      ink3: cssVar("--ink-3"),
      flagBg: cssVar("--flag-bg"),
      paper: cssVar("--paper"),
    };

    // gridlines: rounds
    ctx.strokeStyle = colors.rule;
    ctx.fillStyle = colors.ink3;
    ctx.font = "10px var(--font-archivo), sans-serif";
    ctx.textAlign = "center";
    [1, 2, 3, 5, 8, 12, 16, 20].forEach((r) => {
      const x = xScale(r * 30.75 - 15);
      ctx.beginPath();
      ctx.moveTo(x, M.top);
      ctx.lineTo(x, size.h - M.bottom);
      ctx.stroke();
      ctx.fillText(`R${r}`, x, size.h - M.bottom + 16);
    });
    // y gridlines — 5% steps scaled to the data
    ctx.textAlign = "right";
    const yStep = yMax > 0.5 ? 0.2 : yMax > 0.25 ? 0.1 : 0.05;
    for (let p = 0; p <= yMax + 1e-9; p += yStep) {
      const y = yScale(p);
      ctx.beginPath();
      ctx.moveTo(M.left, y);
      ctx.lineTo(size.w - M.right, y);
      ctx.stroke();
      ctx.fillText(fmtPct(p), M.left - 8, y + 3);
    }

    // axis titles
    ctx.textAlign = "center";
    ctx.fillText("projected draft position →  (earlier picks left)", (size.w + M.left - M.right) / 2, size.h - 8);
    ctx.save();
    ctx.translate(14, (size.h + M.top - M.bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("calibrated MLB probability →", 0, 0);
    ctx.restore();

    // points: draw recessive grades first so elite/high sit on top
    const order: ("low" | "medium" | "high" | "elite")[] = ["low", "medium", "high", "elite"];
    for (const g of order) {
      ctx.fillStyle = colors[g];
      ctx.globalAlpha = g === "low" ? 0.35 : g === "medium" ? 0.5 : 0.9;
      for (const p of pts) {
        if (p.grade !== g) continue;
        const x = xScale(p.proj_pick!);
        const y = yScale(p.mlb_p!);
        ctx.beginPath();
        if (p.type === "hitter") {
          ctx.arc(x, y, g === "elite" || g === "high" ? 3.4 : 2.2, 0, Math.PI * 2);
        } else {
          const s = g === "elite" || g === "high" ? 3.2 : 2.1;
          ctx.rect(x - s, y - s, s * 2, s * 2);
        }
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;

    // hover ring
    if (hover && hover.proj_pick != null && hover.mlb_p != null) {
      const x = xScale(hover.proj_pick);
      const y = yScale(hover.mlb_p);
      ctx.strokeStyle = colors.ruleStrong;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.stroke();
    }

    // brush rect
    if (drag) {
      ctx.strokeStyle = cssVar("--maroon");
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(
        Math.min(drag.x0, drag.x1),
        Math.min(drag.y0, drag.y1),
        Math.abs(drag.x1 - drag.x0),
        Math.abs(drag.y1 - drag.y0),
      );
      ctx.setLineDash([]);
    }
  }, [pts, size, hover, drag, xScale, yScale, yMax]);

  const nearest = useCallback(
    (mx: number, my: number): IndexPlayer | null => {
      let best: IndexPlayer | null = null;
      let bestD = 144; // 12px hit radius
      for (const p of pts) {
        const dx = xScale(p.proj_pick!) - mx;
        const dy = yScale(p.mlb_p!) - my;
        const d = dx * dx + dy * dy;
        if (d < bestD) {
          bestD = d;
          best = p;
        }
      }
      return best;
    },
    [pts, xScale, yScale],
  );

  return (
    <div ref={wrapRef} className="relative w-full">
      <canvas
        ref={canvasRef}
        style={{ width: size.w, height: size.h, cursor: drag ? "crosshair" : "default" }}
        onMouseDown={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - r.left;
          const y = e.clientY - r.top;
          setDrag({ x0: x, y0: y, x1: x, y1: y });
        }}
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - r.left;
          const y = e.clientY - r.top;
          if (drag) {
            setDrag({ ...drag, x1: x, y1: y });
          } else {
            setHover(nearest(x, y));
            setHoverXY([x, y]);
          }
        }}
        onMouseUp={() => {
          if (!drag) return;
          const w = Math.abs(drag.x1 - drag.x0);
          const h = Math.abs(drag.y1 - drag.y0);
          if (w < 8 && h < 8) {
            // click, not brush
            const p = nearest(drag.x0, drag.y0);
            if (p) openDrawer(p.id);
            onBrush(null);
          } else {
            onBrush({
              pickMin: Math.round(xInv(Math.min(drag.x0, drag.x1))),
              pickMax: Math.round(xInv(Math.max(drag.x0, drag.x1))),
              probMin: Number(yInv(Math.max(drag.y0, drag.y1)).toFixed(3)),
              probMax: Number(yInv(Math.min(drag.y0, drag.y1)).toFixed(3)),
            });
          }
          setDrag(null);
        }}
        onMouseLeave={() => {
          setHover(null);
          setDrag(null);
        }}
      />
      {/* quadrant annotation */}
      <div className="pointer-events-none absolute right-8 top-6 max-w-[220px] rounded border border-rule bg-paper-raised/90 p-2 text-[11px] leading-snug text-ink-2">
        <span className="font-semibold text-ink">Upper right = the steals.</span>{" "}
        Projected late, but outcome model says pro. That disagreement is the
        product.
      </div>
      {showPitcherCaution && (
        <div className="pointer-events-none absolute bottom-16 left-14 max-w-[280px] rounded bg-flag-bg px-2 py-1 text-[10px] font-medium leading-snug text-flag">
          Pitcher probabilities are calibrated from a model whose mid-range raw
          scores ran hot — see the tier2 pitcher model card before trusting 15%+.
        </div>
      )}
      {/* tooltip */}
      {hover && !drag && (
        <div
          className="pointer-events-none absolute z-10 w-[230px] rounded border border-rule bg-paper-raised p-2.5 shadow-[var(--shadow-pop)]"
          style={{
            left: Math.min(hoverXY[0] + 14, size.w - 240),
            top: Math.max(8, hoverXY[1] - 70),
          }}
        >
          <div className="flex items-center gap-2">
            <TypeBadge type={hover.type} />
            <span className="truncate text-[13px] font-semibold">{hover.name}</span>
          </div>
          <div className="mt-0.5 text-[11px] text-ink-3">
            {hover.school_abb}
            {hover.conference ? ` · ${hover.conference}` : ""}
          </div>
          <div className="mt-1 flex items-center justify-between text-[12px]">
            <GradeChip grade={hover.grade} small />
            <span>
              pick ~{Math.round(hover.proj_pick!)} · MLB {fmtPct(hover.mlb_p)}
            </span>
          </div>
          <div className="mt-1 text-[10px] text-ink-3">click to open dossier</div>
        </div>
      )}
    </div>
  );
}
