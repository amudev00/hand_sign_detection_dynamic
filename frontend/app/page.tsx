"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Camera,
  Check,
  Settings,
  SlidersHorizontal,
  Hand,
  X,
  Volume2,
  SkipForward,
  Wifi,
  Cpu,
} from "lucide-react";

type PredictionResponse = {
  label: string;
  prob: number;
  combo?: {
    combo: string;
  };
};

type GestureLogItem = {
  label: string;
  time: string;
};

type CalibrationSlot = {
  label: string;
  hint: string;
  image: string | null;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
const PREDICTION_INTERVAL_MS = 700;

const gestureMap = [
  "Media Play",
  "Point Volume",
  "Wave Pointer",
  "Mute",
  "Stop",
  "Thumb Up",
  "Pinch Zoom",
  "Swipe",
  "Rewind",
  "Adjust",
  "Next Track",
  "Select",
  "Pinch",
  "Palm",
  "Hold",
  "Confirm",
];

const calibrationSlotBlueprint: Array<Pick<CalibrationSlot, "label" | "hint">> = [
  { label: "Neutral", hint: "Center your hand" },
  { label: "Open Palm", hint: "Spread fingers fully" },
  { label: "Closed Fist", hint: "Curl fingers inward" },
  { label: "Point", hint: "Index finger forward" },
  { label: "Pinch", hint: "Bring thumb and index together" },
  { label: "Thumbs Up", hint: "Raise thumb clearly" },
];

function getClockLabel(): string {
  const now = new Date();
  return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round(value * 100)));
}

function formatActiveTime(totalSeconds: number): string {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  return [hours, minutes, seconds]
    .map((value) => value.toString().padStart(2, "0"))
    .join(":");
}

function normalizeGestureLabel(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const predictionTimerRef = useRef<number | null>(null);
  const calibrationTimerRef = useRef<number | null>(null);
  const isRequestInFlightRef = useRef(false);
  const calibrationSessionRef = useRef(0);
  const sessionStartedAtRef = useRef(Date.now());

  const [cameraReady, setCameraReady] = useState(false);
  const [detectionRunning, setDetectionRunning] = useState(false);
  const [predictionLabel, setPredictionLabel] = useState("Awaiting Input");
  const [confidence, setConfidence] = useState(0);
  const [comboLabel, setComboLabel] = useState("None");
  const [apiError, setApiError] = useState<string | null>(null);
  const [logItems, setLogItems] = useState<GestureLogItem[]>([]);
  const [calibrationState, setCalibrationState] = useState("Ready");
  const [showCalibration, setShowCalibration] = useState(false);
  const [calibrationStep, setCalibrationStep] = useState<number | null>(null);
  const [calibrationSlots, setCalibrationSlots] = useState<CalibrationSlot[]>(() =>
    calibrationSlotBlueprint.map((slot) => ({ ...slot, image: null })),
  );
  const [pingMs, setPingMs] = useState<number | null>(null);
  const [pingOnline, setPingOnline] = useState(true);
  const [activeSeconds, setActiveSeconds] = useState(0);

  const percent = useMemo(() => clampPercent(confidence), [confidence]);
  const runtimeStatus = useMemo(() => {
    if (!cameraReady) {
      return { label: "Offline", tone: "text-red-300" };
    }
    if (calibrationStep !== null) {
      return { label: "Calibrating", tone: "text-[var(--accent-warn)]" };
    }
    if (detectionRunning) {
      return { label: "Tracking", tone: "text-[var(--accent-green)]" };
    }
    return { label: "Standby", tone: "text-white" };
  }, [calibrationStep, cameraReady, detectionRunning]);
  const normalizedPrediction = useMemo(() => normalizeGestureLabel(predictionLabel), [predictionLabel]);
  const activeGestureIndex = useMemo(
    () => gestureMap.findIndex((item) => normalizeGestureLabel(item).includes(normalizedPrediction) || normalizedPrediction.includes(normalizeGestureLabel(item))),
    [normalizedPrediction],
  );
  const statusSummary = useMemo(() => {
    if (!cameraReady) {
      return "Camera offline";
    }
    if (calibrationStep !== null) {
      return `Calibrating ${calibrationStep + 1}/${calibrationSlotBlueprint.length}`;
    }
    return detectionRunning ? "Detection active" : "Detection paused";
  }, [calibrationStep, cameraReady, detectionRunning]);

  const clearPredictionTimer = useCallback(() => {
    if (predictionTimerRef.current !== null) {
      window.clearInterval(predictionTimerRef.current);
      predictionTimerRef.current = null;
    }
  }, []);

  const clearCalibrationTimer = useCallback(() => {
    if (calibrationTimerRef.current !== null) {
      window.clearTimeout(calibrationTimerRef.current);
      calibrationTimerRef.current = null;
    }
  }, []);

  const captureFrameBlob = useCallback(async (): Promise<Blob | null> => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
      return null;
    }

    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
    }

    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");
    if (!context) {
      return null;
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    return await new Promise<Blob | null>((resolve) => {
      canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.85);
    });
  }, []);

  const captureCalibrationImage = useCallback((): string | null => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
      return null;
    }

    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
    }

    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");
    if (!context) {
      return null;
    }

    context.save();
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    context.restore();

    return canvas.toDataURL("image/jpeg", 0.82);
  }, []);

  const runPrediction = useCallback(async () => {
    if (!detectionRunning || !cameraReady || isRequestInFlightRef.current) {
      return;
    }

    isRequestInFlightRef.current = true;
    try {
      const blob = await captureFrameBlob();
      if (!blob) {
        return;
      }

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Prediction request failed with status ${response.status}`);
      }

      const payload = (await response.json()) as PredictionResponse;
      const nextLabel = payload.label ?? "Unknown";
      const nextConfidence = Number.isFinite(payload.prob) ? payload.prob : 0;

      setPredictionLabel(nextLabel);
      setConfidence(nextConfidence);
      setComboLabel(payload.combo?.combo ?? "None");
      setApiError(null);

      setLogItems((prev) => {
        const nextItem = { label: nextLabel, time: getClockLabel() };
        const deduped = [nextItem, ...prev.filter((item) => item.label !== nextLabel)];
        return deduped.slice(0, 9);
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Prediction failed";
      setApiError(message);
    } finally {
      isRequestInFlightRef.current = false;
    }
  }, [cameraReady, captureFrameBlob, detectionRunning]);

  const stopCamera = useCallback(() => {
    clearPredictionTimer();
    isRequestInFlightRef.current = false;
    setDetectionRunning(false);
    setCameraReady(false);

    const stream = mediaStreamRef.current;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, [clearPredictionTimer]);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1248 },
          height: { ideal: 1248 },
        },
        audio: false,
      });

      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraReady(true);
      setDetectionRunning(true);
      setApiError(null);
    } catch {
      setCameraReady(false);
      setDetectionRunning(false);
      setApiError("Unable to start camera. Check browser permission settings.");
    }
  }, []);

  const clearHistory = useCallback(async () => {
    setLogItems([]);
    setComboLabel("None");
    try {
      await fetch(`${API_BASE_URL}/clear_combos`, { method: "POST" });
      setApiError(null);
    } catch {
      setApiError("Could not clear backend combo history.");
    }
  }, []);

  const runCalibration = useCallback(async () => {
    setShowCalibration(true);
    clearCalibrationTimer();
    calibrationSessionRef.current += 1;
    const sessionId = calibrationSessionRef.current;

    setCalibrationSlots(calibrationSlotBlueprint.map((slot) => ({ ...slot, image: null })));
    setCalibrationStep(null);

    if (!cameraReady) {
      setCalibrationState("Camera required");
      setApiError("Start camera before calibration.");
      return;
    }

    setCalibrationState("Calibrating");
    await clearHistory();
    setApiError(null);

    const captureNext = (index: number) => {
      if (calibrationSessionRef.current !== sessionId) {
        return;
      }

      if (index >= calibrationSlotBlueprint.length) {
        setCalibrationStep(null);
        setCalibrationState("Aligned");
        return;
      }

      setCalibrationStep(index);
      const image = captureCalibrationImage();

      setCalibrationSlots((prev) =>
        prev.map((slot, slotIndex) =>
          slotIndex === index
            ? {
                ...slot,
                image,
              }
            : slot,
        ),
      );

      calibrationTimerRef.current = window.setTimeout(() => {
        captureNext(index + 1);
      }, 450);
    };

    captureNext(0);
  }, [cameraReady, captureCalibrationImage, clearCalibrationTimer, clearHistory]);

  useEffect(() => {
    if (!cameraReady || !detectionRunning) {
      clearPredictionTimer();
      return;
    }

    void runPrediction();
    predictionTimerRef.current = window.setInterval(() => {
      void runPrediction();
    }, PREDICTION_INTERVAL_MS);

    return () => {
      clearPredictionTimer();
    };
  }, [cameraReady, clearPredictionTimer, detectionRunning, runPrediction]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code === "Space") {
        event.preventDefault();
        if (cameraReady) {
          setDetectionRunning(true);
        }
      }
      if (event.key.toLowerCase() === "s") {
        setDetectionRunning(false);
      }
      if (event.key.toLowerCase() === "c") {
        void clearHistory();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [cameraReady, clearHistory]);

  useEffect(() => {
    return () => {
      clearCalibrationTimer();
      calibrationSessionRef.current += 1;
      stopCamera();
    };
  }, [clearCalibrationTimer, stopCamera]);

  useEffect(() => {
    const updateActiveTime = () => {
      setActiveSeconds(Math.floor((Date.now() - sessionStartedAtRef.current) / 1000));
    };

    updateActiveTime();
    const intervalId = window.setInterval(updateActiveTime, 1000);

    return () => {
      window.clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    const updatePing = async () => {
      const startedAt = performance.now();

      try {
        const response = await fetch(`${API_BASE_URL}/clear_combos`, {
          method: "OPTIONS",
          cache: "no-store",
        });

        if (cancelled) {
          return;
        }

        setPingMs(Math.max(1, Math.round(performance.now() - startedAt)));
        setPingOnline(response.ok || response.status < 500);
      } catch {
        if (cancelled) {
          return;
        }

        setPingMs(null);
        setPingOnline(false);
      }
    };

    void updatePing();
    const intervalId = window.setInterval(() => {
      void updatePing();
    }, 5000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, []);

  return (
    <main className="relative flex h-screen flex-col overflow-hidden px-4 py-5 text-[var(--text-main)] md:px-8 md:py-7">
      <div className="mx-auto grid min-h-0 w-full flex-1 max-w-[1440px] grid-cols-1 gap-4 pb-3 lg:grid-cols-[280px_1fr_260px]">

        {/* ──── LEFT: System Overview ──── */}
        <section className="glass hover-panel reveal rounded-3xl p-4 flex flex-col overflow-hidden">
          <h2 className="hud-title text-sm uppercase tracking-[0.2em] text-white/90">System Overview</h2>

          {/* Stat cards */}
          <div className="mt-4 grid grid-cols-2 gap-2">
            <article className="hover-card flex items-center gap-2 rounded-xl border border-[var(--line-soft)] bg-[var(--bg-panel)] px-3 py-2.5">
              <Wifi size={13} className="shrink-0 text-[var(--accent-blue)]" />
              <div>
                <p className="text-[11px] text-[var(--text-dim)]">Ping</p>
                <p className="text-base font-semibold text-white">{pingMs !== null ? `${pingMs}ms` : "Offline"}</p>
              </div>
            </article>
            <article className="hover-card flex items-center gap-2 rounded-xl border border-[var(--line-soft)] bg-[var(--bg-panel)] px-3 py-2.5">
              <Cpu size={13} className="shrink-0 text-[var(--accent-green)]" />
              <div>
                <p className="text-[11px] text-[var(--text-dim)]">Runtime</p>
                <p className={`text-base font-semibold ${runtimeStatus.tone}`}>{runtimeStatus.label}</p>
              </div>
            </article>
          </div>
          <div className="mt-2 flex items-center gap-2 px-1 text-[11px] text-[var(--text-dim)]">
            <span className={`h-2 w-2 rounded-full ${pingOnline ? "bg-emerald-400" : "bg-red-400"}`} />
            <span>{pingOnline ? `${statusSummary} · Backend reachable` : "Backend unreachable"}</span>
          </div>

          {/* Gesture detection */}
          <div className={`hover-card mt-3 rounded-2xl border bg-[var(--bg-panel)] p-3 transition-colors ${cameraReady ? "border-[var(--accent-green)]/25" : "border-[var(--line-soft)]"}`}>
            <p className="hud-title text-[10px] uppercase tracking-[0.24em] text-white/70">Gesture Detection</p>
            <p className="mt-1.5 text-xl font-medium text-white">{predictionLabel}</p>
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-black/40">
              <div
                className="h-full rounded-full bg-gradient-to-r from-[var(--accent-blue)] to-[var(--accent-green)] transition-all duration-500"
                style={{ width: `${percent}%` }}
              />
            </div>
            <div className="mt-1 flex items-center justify-between text-[11px] text-[var(--text-dim)]">
              <span>{percent}% confidence</span>
              <span>{comboLabel !== "None" ? comboLabel : runtimeStatus.label}</span>
            </div>
          </div>

          {/* Command map */}
          <div className="hover-card mt-3 rounded-2xl border border-[var(--line-soft)] bg-[var(--bg-panel)] p-3">
            <div className="mb-2 flex items-center justify-between">
              <p className="hud-title text-[10px] uppercase tracking-[0.2em] text-white/70">Gestural Command Map</p>
              <p className="text-[10px] text-[var(--text-dim)]">16</p>
            </div>
            <div className="grid grid-cols-4 gap-1.5">
              {gestureMap.map((item, index) => (
                <div
                  key={item}
                  className={`hover-tile rounded-xl border p-1.5 text-center text-[10px] leading-tight transition-colors ${
                    activeGestureIndex === index
                      ? "border-[var(--accent-green)]/60 bg-[var(--accent-green)]/12 text-white"
                      : cameraReady
                        ? "border-white/10 bg-[var(--bg-panel-strong)] text-[var(--text-dim)]"
                        : "border-white/8 bg-[var(--bg-panel-strong)]/70 text-white/35"
                  }`}
                >
                  <div className={`mb-0.5 flex justify-center ${activeGestureIndex === index ? "text-[var(--accent-green)]" : "text-white/70"}`}>
                    <Hand size={14} strokeWidth={1.5} />
                  </div>
                  {item}
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ──── CENTER: Camera ──── */}
        <section className="relative reveal hover-panel rounded-[30px] border border-[var(--line)] bg-[var(--bg-panel-strong)] p-2 flex flex-col">
          {/* Camera badge */}
          <div className="absolute left-5 top-5 z-30 flex items-center gap-1.5 rounded-full border border-white/20 bg-black/45 px-3 py-1 text-[11px] font-medium backdrop-blur-sm">
            <span className={`h-2 w-2 rounded-full ${cameraReady ? "bg-emerald-400 pulse-dot" : "bg-red-400"}`} />
            {cameraReady ? "Camera Live" : "Camera Offline"}
          </div>

          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{ transform: "scaleX(-1)" }}
            className={`min-h-0 flex-1 w-full rounded-[26px] object-cover ${cameraReady ? "opacity-100" : "opacity-0 absolute inset-2 w-[calc(100%-16px)] h-[calc(100%-16px)]"}`}
          />

          {/* Offline placeholder */}
          {!cameraReady && (
            <div className="flex flex-1 min-h-0 w-full flex-col items-center justify-center gap-5 rounded-[26px]">
              <div className="relative flex items-center justify-center">
                <div className="absolute h-32 w-32 animate-ping rounded-full border border-[var(--accent-blue)]/20" style={{ animationDuration: "3s" }} />
                <div className="absolute h-20 w-20 rounded-full border border-[var(--accent-blue)]/30" />
                <div className="relative rounded-full border border-[var(--accent-blue)]/50 bg-[var(--accent-blue)]/10 p-6">
                  <Camera size={36} className="text-[var(--accent-blue)]" strokeWidth={1.5} />
                </div>
              </div>
              <div className="text-center">
                <p className="hud-title text-xs uppercase tracking-[0.3em] text-white/40">Camera Feed</p>
                <p className="mt-2 text-sm text-[var(--text-dim)]">Press <span className="font-semibold text-white/70">START CAMERA</span> to begin</p>
              </div>
              <div className="flex items-center gap-2 text-[11px] text-white/20">
                <span className="h-px w-12 bg-white/10" />
                <span>or press Space</span>
                <span className="h-px w-12 bg-white/10" />
              </div>
            </div>
          )}

          <div className="pointer-events-none absolute inset-2 rounded-[28px] border border-white/10" />

          {/* Detection overlay — shifts up when calibration panel is open */}
          <div
            className={`pointer-events-none absolute left-1/2 w-[min(560px,calc(100%-48px))] -translate-x-1/2 rounded-2xl border border-white/15 bg-black/50 px-5 py-4 backdrop-blur-md transition-all duration-300 ${
              showCalibration ? "bottom-[192px]" : "bottom-5"
            }`}
          >
            <p className="hud-title text-[10px] uppercase tracking-[0.22em] text-white/70">Detected Gesture</p>
            <p className="mt-1 text-3xl font-semibold tracking-wide text-white">{predictionLabel}</p>
            <div className="mt-2 h-2 overflow-hidden rounded-full bg-black/50">
              <div
                className="h-full rounded-full bg-gradient-to-r from-[var(--accent-blue)] to-[var(--accent-green)] transition-all duration-500"
                style={{ width: `${percent}%` }}
              />
            </div>
            <div className="mt-1.5 flex items-center justify-between text-xs text-[var(--text-dim)]">
              <span>Confidence: {percent}%</span>
              <span>{comboLabel !== "None" ? `Combo: ${comboLabel}` : runtimeStatus.label}</span>
            </div>
          </div>

          {/* Calibration Hub — toggleable */}
          {showCalibration && (
            <aside className="glass hover-panel reveal absolute bottom-3 left-3 right-3 z-20 rounded-2xl p-3">
              <div className="mb-2 flex items-start justify-between gap-3">
                <div>
                  <p className="hud-title text-sm text-white/85">Calibration Hub</p>
                  <p className="mt-1 text-[11px] text-[var(--text-dim)]">
                    {calibrationStep === null
                      ? calibrationState === "Aligned"
                        ? "Calibration complete. Review the captured samples below."
                        : "Use CALIBRATE to capture pose references from the live feed."
                      : `Capturing ${calibrationSlotBlueprint[calibrationStep]?.label} (${calibrationStep + 1}/${calibrationSlotBlueprint.length})`}
                  </p>
                </div>
                <button
                  type="button"
                  className="hover-icon rounded-full border border-white/20 p-1 text-white/60 transition hover:bg-white/10"
                  onClick={() => {
                    clearCalibrationTimer();
                    calibrationSessionRef.current += 1;
                    setCalibrationStep(null);
                    setShowCalibration(false);
                  }}
                >
                  <X size={12} />
                </button>
              </div>
              <div className="grid grid-cols-3 gap-1.5">
                {calibrationSlots.map((slot, idx) => (
                  <div
                    key={slot.label}
                    className={`hover-tile group relative aspect-video overflow-hidden rounded-xl border bg-gradient-to-br from-white/12 to-white/4 ${
                      calibrationStep === idx
                        ? "border-[var(--accent-green)]/70 shadow-[0_0_0_1px_rgba(65,214,167,0.35)]"
                        : "border-white/12"
                    }`}
                  >
                    {slot.image ? (
                      <img
                        src={slot.image}
                        alt={slot.label}
                        className="h-full w-full object-cover"
                      />
                    ) : null}
                    <div className="absolute inset-0 bg-gradient-to-t from-black/55 via-black/10 to-transparent" />
                    <div className="absolute inset-x-0 bottom-0 p-2">
                      <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-white/90">{slot.label}</p>
                      <p className="text-[11px] text-[var(--text-dim)]">{slot.image ? "Captured" : slot.hint}</p>
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-2 flex items-center justify-between text-[11px] text-[var(--text-dim)]">
                <span>
                  Profile: Alex · Samples: {calibrationSlots.filter((slot) => slot.image).length}/{calibrationSlots.length}
                </span>
                <span className="text-[var(--accent-green)]">{calibrationState}</span>
              </div>
            </aside>
          )}

          {/* Quick controls */}
          <div className="absolute bottom-4 right-4 z-30 flex items-center gap-1 rounded-full border border-white/20 bg-black/50 px-2 py-1.5 backdrop-blur-sm">
            <button
              type="button"
              className="hover-icon rounded-full p-1.5 text-white/70 transition hover:bg-white/10"
              onClick={() => setDetectionRunning((prev) => !prev)}
              title={detectionRunning ? "Pause Detection" : "Resume Detection"}
            >
              <Volume2 size={14} />
            </button>
            <button
              type="button"
              className="hover-icon rounded-full p-1.5 text-white/70 transition hover:bg-white/10"
              onClick={clearHistory}
              title="Clear History"
            >
              <SkipForward size={14} />
            </button>
          </div>
        </section>

        {/* ──── RIGHT: Controls + Log ──── */}
        <section className="flex w-full min-w-0 flex-col gap-3 reveal px-0.5 pb-0.5">
          <button
            type="button"
            onClick={() => {
              if (cameraReady) {
                stopCamera();
              } else {
                void startCamera();
              }
            }}
            className="glass hover-button flex w-full shrink-0 items-center justify-center gap-2.5 rounded-2xl px-4 py-4 text-sm font-semibold text-white transition active:brightness-90"
          >
            <Camera size={17} />
            {cameraReady ? "STOP CAMERA" : "START CAMERA"}
          </button>

          <button
            type="button"
            onClick={() => {
              void runCalibration();
            }}
            className="glass hover-button flex w-full shrink-0 items-center justify-center gap-2.5 rounded-2xl px-4 py-3.5 text-sm font-semibold text-white transition active:brightness-90"
          >
            <SlidersHorizontal size={17} />
            {calibrationStep !== null ? `CALIBRATING ${calibrationStep + 1}/${calibrationSlotBlueprint.length}` : showCalibration ? "RECALIBRATE" : "CALIBRATE"}
          </button>

          <button
            type="button"
            onClick={() => setDetectionRunning((prev) => !prev)}
            className="glass hover-button flex w-full shrink-0 items-center justify-center gap-2.5 rounded-2xl px-4 py-3.5 text-sm font-semibold text-white/80 transition active:brightness-90"
          >
            <Settings size={17} />
            {cameraReady ? (detectionRunning ? "PAUSE DETECTION" : "RESUME DETECTION") : "SETTINGS"}
          </button>

          <div className="glass hover-panel flex flex-col min-h-0 flex-1 rounded-2xl p-4">
            <div className="flex items-center justify-between">
              <h2 className="hud-title text-xs uppercase tracking-[0.18em] text-white/85">Gesture Log</h2>
              <span className="text-[11px] text-[var(--text-dim)]">{logItems.length} entries</span>
            </div>
            <div className="mt-2.5 min-h-0 flex-1 space-y-1.5 overflow-auto pr-0.5">
              {logItems.length === 0 ? (
                <p className="hover-card rounded-xl border border-white/10 bg-black/20 px-3 py-2.5 text-xs text-[var(--text-dim)]">
                  No gestures captured yet.
                </p>
              ) : (
                logItems.map((item) => (
                  <div
                    key={`${item.label}-${item.time}`}
                    className="hover-card flex items-center justify-between rounded-xl border border-white/10 bg-black/20 px-3 py-2"
                  >
                    <span className="text-xs text-white/85">{item.label}</span>
                    <span className="text-[11px] text-[var(--text-dim)]">{item.time}</span>
                  </div>
                ))
              )}
            </div>
            <div className="mt-2.5 flex items-center gap-1.5 text-[11px] text-[var(--text-dim)]">
              <Check size={12} />
              <span>Space: start · S: stop · C: clear</span>
            </div>
          </div>

          {apiError ? (
            <div className="hover-card rounded-xl border border-[var(--accent-warn)]/60 bg-[var(--accent-warn)]/10 px-3 py-2 text-xs text-amber-200">
              {apiError}
            </div>
          ) : null}

          {comboLabel && comboLabel !== "None" && (
            <div className="hover-card rounded-xl border border-[var(--accent-green)]/40 bg-[var(--accent-green)]/10 px-3 py-2 text-sm text-[var(--accent-green)]">
              Combo: <span className="font-semibold">{comboLabel}</span>
            </div>
          )}
        </section>

      </div>
      <div className="mx-auto flex w-full max-w-[1440px] items-center justify-between gap-3 border-t border-white/10 px-1 pt-2 text-[11px] text-[var(--text-dim)]">
        <span className="truncate">{pingOnline ? `Backend ${pingMs !== null ? `${pingMs}ms` : "online"}` : "Backend offline"}</span>
        <span className="truncate">{runtimeStatus.label}</span>
        <span className="truncate">{comboLabel !== "None" ? `Combo ${comboLabel}` : predictionLabel}</span>
        <span className="font-medium text-white/80">{formatActiveTime(activeSeconds)}</span>
      </div>
    </main>
  );
}
