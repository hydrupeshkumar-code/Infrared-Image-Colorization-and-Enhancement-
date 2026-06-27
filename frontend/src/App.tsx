import { useEffect, useRef, useState } from "react";
import {
  asset, getJob, InferError, postInfer,
  type JobRecord, type JobStatus,
} from "./api";
import Nav from "./components/Nav";
import Hero from "./components/Hero";
import ResultsDashboard from "./components/ResultsDashboard";
import { PLACEHOLDER_INPUT, PLACEHOLDER_OUTPUT } from "./placeholders";

type UiStatus = JobStatus | "idle";

export default function App() {
  const [pct, setPct] = useState(50);
  const [status, setStatus] = useState<UiStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [record, setRecord] = useState<JobRecord | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const openPicker = () => fileInputRef.current?.click();

  // Poll the job until it reaches a terminal state. React state only.
  useEffect(() => {
    if (!jobId) return;
    let active = true;
    const tick = async () => {
      try {
        const rec = await getJob(jobId);
        if (!active) return;
        setRecord(rec);
        setStatus(rec.status);
        if (rec.status === "failed") setError(rec.error || "Inference failed.");
        if (rec.status === "done" || rec.status === "failed") {
          clearInterval(timer);
        }
      } catch (e) {
        if (active) setError((e as Error).message);
      }
    };
    const timer = setInterval(tick, 1500);
    tick();
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [jobId]);

  const onFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow re-selecting the same file
    if (!file) return;
    setError(null);
    setRecord(null);
    setJobId(null);
    setStatus("queued");
    try {
      const id = await postInfer(file);
      setJobId(id);
    } catch (err) {
      setStatus("idle");
      setError(
        err instanceof InferError ? err.message : (err as Error).message
      );
    }
  };

  const done = record?.status === "done" && record.artifacts;
  const beforeSrc = done ? asset(record!.artifacts!.input_preview_png) : PLACEHOLDER_INPUT;
  const afterSrc = done ? asset(record!.artifacts!.rgb_preview_png) : PLACEHOLDER_OUTPUT;

  return (
    <div
      className="min-h-screen bg-black tracking-[-0.02em]"
      style={{ fontFamily: "Inter, sans-serif" }}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".tif,.tiff"
        className="hidden"
        onChange={onFileChange}
      />

      <Nav onUpload={openPicker} />

      <Hero
        beforeSrc={beforeSrc}
        afterSrc={afterSrc}
        pct={pct}
        onPct={setPct}
        onUpload={openPicker}
        status={status}
        error={error}
      />

      {done && (
        <ResultsDashboard artifacts={record!.artifacts!} metrics={record!.metrics} />
      )}
    </div>
  );
}
