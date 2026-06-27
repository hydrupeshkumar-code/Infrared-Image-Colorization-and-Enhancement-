import { Download } from "lucide-react";
import { asset, type Artifacts, type Metrics } from "../api";

interface Props {
  artifacts: Artifacts;
  metrics: Metrics | null;
}

interface PanelDef {
  src: string;
  title: string;
  caption: string;
  emphasize?: boolean;
}

function fmt(v: number | null, digits: number, suffix = ""): string {
  return v == null ? "—" : `${v.toFixed(digits)}${suffix}`;
}

function MetricCard({
  label, value, emphasize = false,
}: { label: string; value: string; emphasize?: boolean }) {
  return (
    <div
      className={
        "rounded-xl p-4 flex flex-col gap-1 " +
        (emphasize
          ? "bg-[#e8702a]/10 border-2 border-[#e8702a]/60"
          : "bg-white/5 border border-white/10")
      }
    >
      <span className="text-xs uppercase tracking-wide text-white/50">{label}</span>
      <span className={emphasize ? "text-2xl font-semibold text-[#f3935a]" : "text-xl font-semibold text-white"}>
        {value}
      </span>
    </div>
  );
}

export default function ResultsDashboard({ artifacts, metrics }: Props) {
  const panels: PanelDef[] = [
    { src: artifacts.input_preview_png, title: "Input · LR TIR 200m",
      caption: "The single thermal band you uploaded." },
    { src: artifacts.sr_preview_png, title: "SR · TIR 100m",
      caption: "2× super-resolved, same thermal colormap & scale." },
    { src: artifacts.rgb_preview_png, title: "Predicted RGB 100m",
      caption: "Colorized translation of the thermal field." },
    { src: artifacts.residual_preview_png, title: "Residual map",
      caption: "Invented structure (should be near zero).", emphasize: true },
  ];

  return (
    <section className="relative bg-black px-5 sm:px-10 md:px-14 py-16 sm:py-24">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-white text-3xl sm:text-4xl font-playfair italic mb-2">
          Faithfulness audit
        </h2>
        <p className="text-white/60 text-sm mb-10 max-w-2xl">
          The residual map and Kelvin-unit metrics exist to prove the model sharpens
          what is present and does not hallucinate new structure.
        </p>

        {/* 4-panel grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-12">
          {panels.map((p) => (
            <div
              key={p.title}
              className={
                "rounded-xl p-3 " +
                (p.emphasize
                  ? "bg-white/5 ring-2 ring-[#e8702a]/70 border border-[#e8702a]/40"
                  : "bg-white/5 border border-white/10")
              }
            >
              <img
                src={asset(p.src)}
                alt={p.title}
                className="w-full aspect-square object-cover rounded-lg bg-black"
                style={{ imageRendering: "pixelated" }}
              />
              <div className="mt-3">
                <p className={"text-sm font-medium " + (p.emphasize ? "text-[#f3935a]" : "text-white")}>
                  {p.title}
                </p>
                <p className="text-xs text-white/50">{p.caption}</p>
                {p.emphasize && (
                  <p className="text-[11px] text-white/40 mt-2 leading-relaxed">
                    Diverging colormap (RdBu_r) centered at 0: blue = SR below input,
                    red = SR above input. Flat / pale ≈ faithful.
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Metrics row */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-10">
          <MetricCard label="PSNR (SR)" value={fmt(metrics?.psnr_sr ?? null, 2, " dB")} />
          <MetricCard label="SSIM (SR)" value={fmt(metrics?.ssim_sr ?? null, 3)} />
          <MetricCard label="PSNR (RGB)" value={fmt(metrics?.psnr_rgb ?? null, 2, " dB")} />
          <MetricCard label="SSIM (RGB)" value={fmt(metrics?.ssim_rgb ?? null, 3)} />
          <MetricCard label="SR mean-bias" value={fmt(metrics?.sr_mean_bias_k ?? null, 3, " K")} emphasize />
          <MetricCard label="SR RMSE" value={fmt(metrics?.sr_rmse_k ?? null, 3, " K")} emphasize />
        </div>
        <p className="text-xs text-white/40 mb-10 max-w-2xl">
          PSNR/SSIM require an HR ground-truth reference and read “—” at inference —
          we never fabricate them. The two Kelvin metrics are the physics-fidelity
          claim: they measure how well the super-resolved output collapses back to
          the observed 200m input.
        </p>

        {/* Downloads */}
        <div className="flex flex-wrap gap-3">
          <a
            href={asset(artifacts.sr_tif)}
            download
            className="flex items-center gap-2 bg-white/10 hover:bg-white/20 text-white text-sm font-medium px-5 py-2.5 rounded-full transition-colors border border-white/10"
          >
            <Download size={16} /> SR TIR (GeoTIFF)
          </a>
          <a
            href={asset(artifacts.rgb_tif)}
            download
            className="flex items-center gap-2 bg-white/10 hover:bg-white/20 text-white text-sm font-medium px-5 py-2.5 rounded-full transition-colors border border-white/10"
          >
            <Download size={16} /> RGB (GeoTIFF)
          </a>
        </div>
      </div>
    </section>
  );
}
