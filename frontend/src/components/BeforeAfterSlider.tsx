import { useCallback, useEffect, useRef } from "react";
import { ChevronsLeftRight } from "lucide-react";

interface Props {
  /** Top layer, clipped to the left `pct`% — the INPUT TIR preview. */
  beforeSrc: string;
  /** Bottom layer — the OUTPUT RGB preview. */
  afterSrc: string;
  pct: number;
  onPct: (pct: number) => void;
}

export default function BeforeAfterSlider({ beforeSrc, afterSrc, pct, onPct }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const setFromClientX = useCallback(
    (clientX: number) => {
      const el = containerRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const next = ((clientX - rect.left) / rect.width) * 100;
      onPct(Math.max(0, Math.min(100, next)));
    },
    [onPct]
  );

  // Global listeners while dragging so the grip keeps tracking outside the handle.
  useEffect(() => {
    const move = (e: MouseEvent | TouchEvent) => {
      if (!dragging.current) return;
      const x = "touches" in e ? e.touches[0]?.clientX : (e as MouseEvent).clientX;
      if (x != null) {
        setFromClientX(x);
        if ("touches" in e) e.preventDefault();
      }
    };
    const up = () => {
      dragging.current = false;
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    window.addEventListener("touchmove", move, { passive: false });
    window.addEventListener("touchend", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      window.removeEventListener("touchmove", move);
      window.removeEventListener("touchend", up);
    };
  }, [setFromClientX]);

  const start = (clientX: number) => {
    dragging.current = true;
    setFromClientX(clientX);
  };

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 select-none"
      onMouseDown={(e) => start(e.clientX)}
      onTouchStart={(e) => start(e.touches[0].clientX)}
      style={{ touchAction: "none", cursor: "ew-resize" }}
    >
      {/* Bottom layer: OUTPUT RGB */}
      <div
        className="absolute inset-0 z-10 bg-cover bg-center"
        style={{ backgroundImage: `url("${afterSrc}")` }}
      />
      {/* Top layer: INPUT TIR, clipped to the left pct% */}
      <div
        className="absolute inset-0 z-20 bg-cover bg-center"
        style={{
          backgroundImage: `url("${beforeSrc}")`,
          clipPath: `inset(0 ${100 - pct}% 0 0)`,
        }}
      />

      {/* Badges */}
      <span className="absolute top-20 left-5 z-30 uppercase text-xs tracking-wide bg-black/40 backdrop-blur text-white rounded px-3 py-1">
        Input · TIR 200m
      </span>
      <span className="absolute top-20 right-5 z-30 uppercase text-xs tracking-wide bg-black/40 backdrop-blur text-white rounded px-3 py-1">
        Output · RGB 100m
      </span>

      {/* Handle */}
      <div
        className="absolute top-0 bottom-0 z-40"
        style={{ left: `${pct}%`, transform: "translateX(-50%)" }}
      >
        <div className="absolute top-0 bottom-0 left-1/2 -translate-x-1/2 w-0.5 bg-white" />
        <button
          aria-label="Drag to compare"
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-11 w-11 rounded-full bg-white text-gray-900 shadow-lg flex items-center justify-center"
          onMouseDown={(e) => {
            e.stopPropagation();
            start(e.clientX);
          }}
          onTouchStart={(e) => {
            e.stopPropagation();
            start(e.touches[0].clientX);
          }}
        >
          <ChevronsLeftRight size={20} />
        </button>
      </div>
    </div>
  );
}
