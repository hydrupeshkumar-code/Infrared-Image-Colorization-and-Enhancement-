import { Loader2, AlertTriangle } from "lucide-react";
import type { JobStatus } from "../api";
import BeforeAfterSlider from "./BeforeAfterSlider";

interface Props {
  beforeSrc: string;
  afterSrc: string;
  pct: number;
  onPct: (pct: number) => void;
  onUpload: () => void;
  status: JobStatus | "idle";
  error: string | null;
  notReady?: boolean;
}

export default function Hero({
  beforeSrc, afterSrc, pct, onPct, onUpload, status, error, notReady = false,
}: Props) {
  const busy = status === "queued" || status === "running";

  return (
    <section
      className="relative w-full overflow-hidden min-h-screen bg-black"
      style={{ minHeight: "100dvh" }}
    >
      {/* Slider centerpiece */}
      <div className="absolute inset-0 hero-zoom">
        <BeforeAfterSlider beforeSrc={beforeSrc} afterSrc={afterSrc} pct={pct} onPct={onPct} />
      </div>

      {/* Heading */}
      <div className="z-50 absolute top-[12%] inset-x-0 flex flex-col items-center text-center px-5 pointer-events-none">
        <h1 className="text-white leading-[0.95]">
          <span
            className="block font-playfair italic font-normal text-5xl sm:text-7xl md:text-8xl hero-anim hero-reveal"
            style={{ letterSpacing: "-0.05em", animationDelay: "0.25s" }}
          >
            Thermal made
          </span>
          <span
            className="block font-normal text-5xl sm:text-7xl md:text-8xl -mt-1 hero-anim hero-reveal"
            style={{ letterSpacing: "-0.08em", animationDelay: "0.42s" }}
          >
            visible &amp; sharp
          </span>
        </h1>
      </div>

      {/* Bottom-left blurb */}
      <div
        className="z-50 hidden sm:block absolute bottom-14 left-10 md:left-14 max-w-[280px] hero-anim hero-fade"
        style={{ animationDelay: "0.7s" }}
      >
        <p className="text-sm text-white/80 leading-relaxed">
          A single 200m thermal band, super-resolved to 100m and colorized into RGB,
          with physics-consistency constraints that keep every pixel faithful to the input.
        </p>
      </div>

      {/* Bottom-right blurb + upload */}
      <div
        className="z-50 absolute bottom-10 sm:bottom-24 left-5 right-5 sm:left-auto sm:right-10 md:right-14 max-w-full sm:max-w-[280px] flex flex-col items-start gap-4 sm:gap-5 hero-anim hero-fade"
        style={{ animationDelay: "0.85s" }}
      >
        <p className="text-xs sm:text-sm text-white/80 leading-relaxed">
          Drag to compare. Residual maps and Kelvin-unit metrics audit the result,
          so the model sharpens what's there and never invents.
        </p>

        <button
          onClick={onUpload}
          disabled={busy || notReady}
          className="bg-[#e8702a] hover:bg-[#d2611f] text-white text-sm font-medium px-7 py-3 rounded-full transition-all hover:scale-[1.03] active:scale-95 hover:shadow-lg hover:shadow-[#e8702a]/30 disabled:opacity-60 disabled:hover:scale-100 flex items-center gap-2"
        >
          {busy && <Loader2 size={16} className="spin" />}
          {busy ? (status === "queued" ? "Queued…" : "Running…") : "Upload & Run"}
        </button>

        {notReady && (
          <div className="flex items-start gap-2 text-xs text-amber-200 bg-amber-900/40 border border-amber-500/40 rounded-lg px-3 py-2 max-w-full">
            <AlertTriangle size={16} className="mt-0.5 shrink-0" />
            <span className="break-words">
              Backend has no trained model checkpoints yet — run{" "}
              <code className="font-mono">make serve</code> (or{" "}
              <code className="font-mono">make smoke</code>) before uploading.
            </span>
          </div>
        )}

        {status === "failed" && error && (
          <div className="flex items-start gap-2 text-xs text-red-200 bg-red-900/40 border border-red-500/40 rounded-lg px-3 py-2 max-w-full">
            <AlertTriangle size={16} className="mt-0.5 shrink-0" />
            <span className="break-words">{error}</span>
          </div>
        )}
        {status === "idle" && error && (
          <div className="flex items-start gap-2 text-xs text-red-200 bg-red-900/40 border border-red-500/40 rounded-lg px-3 py-2 max-w-full">
            <AlertTriangle size={16} className="mt-0.5 shrink-0" />
            <span className="break-words">{error}</span>
          </div>
        )}
      </div>
    </section>
  );
}
