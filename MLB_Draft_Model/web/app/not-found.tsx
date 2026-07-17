import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto flex max-w-[600px] flex-col items-center px-4 py-20 text-center">
      <div
        className="text-[64px] font-semibold leading-none text-text-disabled"
        style={{ fontFamily: "var(--font-fraunces)" }}
      >
        404
      </div>
      <p className="mt-4 text-text-secondary">
        This page doesn&apos;t exist. The model can&apos;t project it either.
      </p>
      <Link
        href="/board"
        className="mt-6 rounded-sm bg-accent px-4 py-2 text-[13px] font-semibold text-surface-2 hover:bg-accent-hover"
      >
        Back to the board
      </Link>
    </div>
  );
}
