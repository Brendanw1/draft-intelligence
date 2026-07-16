import type { Metadata } from "next";
import { Archivo, Fraunces } from "next/font/google";
import "./globals.css";
import { TopBar } from "@/components/chrome/TopBar";
import { CompareTray } from "@/components/chrome/CompareTray";
import { GlobalDrawer } from "@/components/player/GlobalDrawer";

const archivo = Archivo({
  subsets: ["latin"],
  variable: "--font-archivo",
  display: "swap",
});

const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-fraunces",
  display: "swap",
  axes: ["opsz"],
});

export const metadata: Metadata = {
  title: "VT Draft Intelligence",
  description:
    "College-to-pro draft model workspace — 2026 projections, calibrated MLB probabilities, model transparency.",
  robots: { index: false, follow: false },
};

const themeInit = `(function(){try{var t=localStorage.getItem('vtdi-theme');if(t==='dark'||(!t&&window.matchMedia('(prefers-color-scheme: dark)').matches)){document.documentElement.dataset.theme='dark'}}catch(e){}})()`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInit }} />
      </head>
      <body className={`${archivo.variable} ${fraunces.variable} min-h-screen`}>
        <TopBar />
        <main className="pb-16">{children}</main>
        <CompareTray />
        <GlobalDrawer />
      </body>
    </html>
  );
}
