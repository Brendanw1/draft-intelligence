"use client";

import { create } from "zustand";

interface UIState {
  drawerId: string | null;
  compareIds: string[];
  openDrawer: (id: string) => void;
  closeDrawer: () => void;
  toggleCompare: (id: string) => void;
  clearCompare: () => void;
}

export const useUI = create<UIState>((set) => ({
  drawerId: null,
  compareIds: [],
  openDrawer: (id) => set({ drawerId: id }),
  closeDrawer: () => set({ drawerId: null }),
  toggleCompare: (id) =>
    set((s) => ({
      compareIds: s.compareIds.includes(id)
        ? s.compareIds.filter((x) => x !== id)
        : s.compareIds.length >= 4
          ? s.compareIds
          : [...s.compareIds, id],
    })),
  clearCompare: () => set({ compareIds: [] }),
}));

// Local-first scouting notes — never model input.
export function loadNote(id: string): string {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem(`vtdi-note-${id}`) ?? "";
}

export function saveNote(id: string, text: string) {
  if (typeof window === "undefined") return;
  if (text) window.localStorage.setItem(`vtdi-note-${id}`, text);
  else window.localStorage.removeItem(`vtdi-note-${id}`);
}
