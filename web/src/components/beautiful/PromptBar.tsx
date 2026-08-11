"use client";

import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { MODEL_META, SAMPLE_REVIEWS, type ModelKey } from "@/lib/types";

/* Beautiful UI Prompt Bar — adapted for ReelTone (no glimm / autoplay) */

function Icon({
  children,
  size = 15,
  strokeWidth = 1.8,
}: {
  children: ReactNode;
  size?: number;
  strokeWidth?: number;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

const GLYPHS = {
  clip: (
    <path d="m21.4 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48" />
  ),
  spark: <path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M18.4 5.6l-2.8 2.8M8.4 15.6l-2.8 2.8" />,
};

type Source = {
  key: string;
  name: string;
  desc: string;
  text?: string;
  attach?: boolean;
};

const SOURCES: Source[] = [
  {
    key: "attach",
    name: "Paste sample",
    desc: "Load a canned review",
    attach: true,
  },
  ...SAMPLE_REVIEWS.map((s) => ({
    key: s.id,
    name: s.label,
    desc: s.text.slice(0, 48) + "…",
    text: s.text,
  })),
];

const COMMANDS = [
  { key: "analyze", name: "/analyze", desc: "Run sentiment prediction" },
  { key: "compare", name: "/compare", desc: "Compare all trained models" },
  { key: "clear", name: "/clear", desc: "Clear the review box" },
];

function parseToken(
  draft: string,
): { kind: "at" | "slash"; query: string; start: number } | null {
  const match = /(^|\s)([@/])([\w-]*)$/.exec(draft);
  if (!match) return null;
  return {
    kind: match[2] === "@" ? "at" : "slash",
    query: match[3].toLowerCase(),
    start: match.index + match[1].length,
  };
}

export default function PromptBar({
  value,
  onChange,
  model,
  onModelChange,
  models,
  onSubmit,
  onCompare,
  disabled = false,
  variant = "Rounded",
}: {
  value: string;
  onChange: (next: string) => void;
  model: ModelKey;
  onModelChange: (next: ModelKey) => void;
  models: { key: ModelKey; available: boolean }[];
  onSubmit: () => void;
  onCompare?: () => void;
  disabled?: boolean;
  variant?: string;
}) {
  const pill = variant === "Pill";
  const [dismissed, setDismissed] = useState(false);
  const [plusOpen, setPlusOpen] = useState(false);
  const [modelOpen, setModelOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [expanded, setExpanded] = useState(false);
  const [rowBox, setRowBox] = useState<{ top: number; height: number } | null>(
    null,
  );
  const [engaged, setEngaged] = useState(false);
  const [modelBox, setModelBox] = useState<{
    top: number;
    height: number;
  } | null>(null);
  const [modelHovered, setModelHovered] = useState<number | null>(null);
  const controlsRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const measureRef = useRef<HTMLSpanElement>(null);
  const modelRef = useRef<HTMLButtonElement>(null);
  const rowRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const modelRowRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const selectedMeta = MODEL_META[model];
  const token = dismissed ? null : parseToken(value);
  const menu: "at" | "slash" | null = plusOpen ? "at" : (token?.kind ?? null);
  const query = plusOpen ? "" : (token?.query ?? "");

  const rows: { key: string; name: string; desc: string }[] =
    menu === "at"
      ? SOURCES.filter((s) => s.name.toLowerCase().includes(query))
      : menu === "slash"
        ? COMMANDS.filter((c) => c.name.slice(1).startsWith(query))
        : [];

  const modelIndex = Math.max(
    0,
    models.findIndex((m) => m.key === model),
  );

  useLayoutEffect(() => {
    setActive(0);
    setEngaged(false);
  }, [menu, query]);

  useLayoutEffect(() => {
    const target = rowRefs.current[active];
    if (target) setRowBox({ top: target.offsetTop, height: target.offsetHeight });
  }, [menu, query, active, rows.length]);

  useLayoutEffect(() => {
    if (!modelOpen) return;
    const target = modelRowRefs.current[modelHovered ?? modelIndex];
    if (target)
      setModelBox({ top: target.offsetTop, height: target.offsetHeight });
  }, [modelOpen, modelHovered, modelIndex]);

  useLayoutEffect(() => {
    const input = inputRef.current;
    const controls = controlsRef.current;
    const measure = measureRef.current;
    const modelButton = modelRef.current;
    if (!input || !controls || !measure || !modelButton) return;

    const fixedControlsWidth = 28 * 2 + modelButton.offsetWidth;
    const inlineGaps = 4 * 3;
    const inlineInputWidth =
      controls.clientWidth - fixedControlsWidth - inlineGaps;
    const needsFullWidth =
      value.includes("\n") || measure.offsetWidth + 8 > inlineInputWidth;
    if (needsFullWidth !== expanded) setExpanded(needsFullWidth);

    const minHeight = 28;
    const maxHeight = 140;
    input.style.height = "0px";
    const contentHeight = input.scrollHeight;
    input.style.height = `${Math.min(Math.max(contentHeight, minHeight), maxHeight)}px`;
    input.style.overflowY = contentHeight > maxHeight ? "auto" : "hidden";
  }, [value, expanded]);

  const closeMenus = () => {
    setPlusOpen(false);
    setModelOpen(false);
  };

  const applySample = (text: string) => {
    onChange(text);
    setPlusOpen(false);
    setDismissed(false);
    inputRef.current?.focus();
  };

  const pick = (row: { key: string; name: string }) => {
    if (menu === "slash") {
      const prefix = token ? value.slice(0, token.start) : value;
      if (row.key === "clear") {
        onChange("");
      } else if (row.key === "compare") {
        onChange(prefix.trimEnd());
        onCompare?.();
      } else if (row.key === "analyze") {
        onChange(prefix.trimEnd());
        onSubmit();
      }
      setPlusOpen(false);
      setDismissed(false);
      inputRef.current?.focus();
      return;
    }

    const source = SOURCES.find((s) => s.key === row.key);
    if (source?.attach) {
      applySample(SAMPLE_REVIEWS[0].text);
      return;
    }
    if (source?.text) {
      applySample(source.text);
      return;
    }
    onChange(`${token ? value.slice(0, token.start) : value}@${row.name} `);
    setPlusOpen(false);
  };

  const canSend = value.trim().length > 0 && !disabled;

  return (
    <div className="relative w-full">
      {menu && (
        <div
          onMouseLeave={() => setEngaged(false)}
          className="absolute inset-x-0 bottom-full z-10 mb-2 rounded-[10px] bg-surface p-1 shadow-raised"
          style={{
            animation: "pop-in 180ms cubic-bezier(0.23,1,0.32,1) both",
            transformOrigin: "bottom center",
          }}
        >
          <span
            aria-hidden
            className="pointer-events-none absolute inset-x-1 rounded-[6px] bg-hover"
            style={{
              top: rowBox?.top ?? 0,
              height: rowBox?.height ?? 0,
              opacity: rowBox && engaged && rows.length > 0 ? 1 : 0,
              transition:
                "top 220ms cubic-bezier(0.23,1,0.32,1), height 220ms cubic-bezier(0.23,1,0.32,1), opacity 150ms ease",
            }}
          />
          {rows.map((row, i) => {
            const source =
              menu === "at" ? SOURCES.find((s) => s.key === row.key) : undefined;
            return (
              <button
                key={row.key}
                type="button"
                ref={(el) => {
                  rowRefs.current[i] = el;
                }}
                onMouseDown={(event) => event.preventDefault()}
                onMouseEnter={() => {
                  setActive(i);
                  setEngaged(true);
                }}
                onClick={() => pick(row)}
                className="relative z-10 flex h-9 w-full items-center gap-2.5 rounded-[6px] px-2 text-left"
              >
                {source && (
                  <span className="flex size-[22px] shrink-0 items-center justify-center text-ink-2">
                    <Icon size={15}>
                      {source.attach ? GLYPHS.clip : GLYPHS.spark}
                    </Icon>
                  </span>
                )}
                <span className="shrink-0 text-[12.5px] font-medium text-ink">
                  {row.name}
                </span>
                <span className="min-w-0 flex-1 truncate text-[12px] text-ink-3">
                  {row.desc}
                </span>
              </button>
            );
          })}
          {rows.length === 0 && (
            <div className="flex h-9 items-center px-2 text-[12px] text-ink-3">
              No matches for “{query}”
            </div>
          )}
          <div className="mt-1 border-t border-line px-2 pt-1.5 pb-1 text-[11px] text-ink-3">
            {menu === "at"
              ? "Type @ for sample reviews"
              : "Type / for commands"}
          </div>
        </div>
      )}

      {modelOpen && (
        <div
          onMouseLeave={() => setModelHovered(null)}
          className="absolute right-0 bottom-full z-10 mb-2 w-52 rounded-[10px] bg-surface p-1 shadow-raised"
          style={{
            animation: "pop-in 180ms cubic-bezier(0.23,1,0.32,1) both",
            transformOrigin: "bottom right",
          }}
        >
          <span
            aria-hidden
            className="pointer-events-none absolute inset-x-1 rounded-[6px] bg-hover"
            style={{
              top: modelBox?.top ?? 0,
              height: modelBox?.height ?? 0,
              opacity: modelBox && modelHovered !== null ? 1 : 0,
              transition:
                "top 220ms cubic-bezier(0.23,1,0.32,1), height 220ms cubic-bezier(0.23,1,0.32,1), opacity 150ms ease",
            }}
          />
          {models.map((m, i) => {
            const meta = MODEL_META[m.key];
            return (
              <button
                key={m.key}
                type="button"
                ref={(el) => {
                  modelRowRefs.current[i] = el;
                }}
                onMouseDown={(event) => event.preventDefault()}
                onMouseEnter={() => setModelHovered(i)}
                onClick={() => {
                  onModelChange(m.key);
                  setModelOpen(false);
                  inputRef.current?.focus();
                }}
                className="relative z-10 flex h-[30px] w-full items-center gap-2 rounded-[6px] px-2 text-left"
              >
                <span className="min-w-0 flex-1 truncate text-[12.5px] font-medium text-ink">
                  {meta.label}
                </span>
                <span className="shrink-0 text-[11px] text-ink-3">
                  {m.available ? meta.blurb : "Not trained"}
                </span>
                <span
                  className={`shrink-0 text-ink ${m.key === model ? "" : "invisible"}`}
                >
                  <Icon size={13} strokeWidth={2.5}>
                    <path d="M20 6L9 17l-5-5" />
                  </Icon>
                </span>
              </button>
            );
          })}
        </div>
      )}

      <div
        className={`relative isolate flex flex-col gap-1.5 overflow-hidden border border-line bg-surface p-1.5 shadow-card transition-[border-color,border-radius] duration-150 focus-within:border-line-strong ${
          pill
            ? expanded
              ? "rounded-[24px]"
              : "rounded-full"
            : "rounded-[14px]"
        }`}
      >
        <span
          ref={measureRef}
          aria-hidden="true"
          className="pointer-events-none absolute invisible whitespace-pre text-[13px] leading-[18px]"
        >
          {value}
        </span>

        <div
          ref={controlsRef}
          className={`grid items-end gap-x-1 gap-y-1.5 ${
            expanded
              ? "grid-cols-[minmax(0,1fr)_auto_28px]"
              : "grid-cols-[28px_minmax(0,1fr)_auto_28px]"
          }`}
        >
          <button
            type="button"
            aria-label="Add samples and commands"
            aria-expanded={plusOpen}
            onClick={() => {
              setModelOpen(false);
              setPlusOpen((current) => !current);
              inputRef.current?.focus();
            }}
            className={`flex size-7 shrink-0 items-center justify-center justify-self-start text-ink-3 transition-[background-color,color,transform] duration-150 hover:bg-hover hover:text-ink active:scale-[0.94] ${
              pill ? "rounded-full" : "rounded-[8px]"
            } ${plusOpen ? "bg-hover text-ink" : ""} ${
              expanded ? "col-start-1 row-start-2" : "col-start-1 row-start-1"
            }`}
          >
            <Icon size={16} strokeWidth={2}>
              <path d="M12 5v14M5 12h14" />
            </Icon>
          </button>

          <textarea
            ref={inputRef}
            rows={1}
            value={value}
            disabled={disabled}
            onChange={(event) => {
              onChange(event.target.value.slice(0, 5000));
              setDismissed(false);
              setPlusOpen(false);
            }}
            onKeyDown={(event) => {
              if (menu && rows.length > 0) {
                if (event.key === "ArrowDown" || event.key === "ArrowUp") {
                  event.preventDefault();
                  setEngaged(true);
                  setActive(
                    (current) =>
                      (current +
                        (event.key === "ArrowDown" ? 1 : rows.length - 1)) %
                      rows.length,
                  );
                  return;
                }
                if (
                  (event.key === "Enter" && !event.shiftKey) ||
                  event.key === "Tab"
                ) {
                  event.preventDefault();
                  pick(rows[active]);
                  return;
                }
              }
              if (event.key === "Escape") {
                setDismissed(true);
                closeMenus();
                return;
              }
              if (
                event.key === "Enter" &&
                !event.shiftKey &&
                !event.nativeEvent.isComposing
              ) {
                event.preventDefault();
                if (canSend) onSubmit();
              }
            }}
            placeholder="Paste a movie review… (@ samples, / commands)"
            aria-label="Movie review"
            className={`min-h-7 min-w-0 w-full resize-none bg-transparent px-1 py-[5px] text-[13px] leading-[18px] text-ink outline-none [overflow-wrap:anywhere] placeholder:text-ink-3 disabled:opacity-50 ${
              expanded
                ? "col-span-full col-start-1 row-start-1"
                : "col-start-2 row-start-1"
            }`}
          />

          <button
            ref={modelRef}
            type="button"
            aria-expanded={modelOpen}
            aria-label="Choose model"
            onClick={() => {
              setPlusOpen(false);
              setModelOpen((current) => !current);
            }}
            className={`flex h-7 shrink-0 items-center gap-1 px-1.5 text-[12px] font-medium text-ink-2 transition-colors duration-150 hover:bg-hover hover:text-ink ${
              pill ? "rounded-full" : "rounded-[8px]"
            } ${expanded ? "col-start-2 row-start-2" : "col-start-3 row-start-1"}`}
          >
            {selectedMeta.label}
            <span className="text-ink-3">
              <Icon size={11} strokeWidth={2.4}>
                <path d="M6 9l6 6 6-6" />
              </Icon>
            </span>
          </button>

          <button
            type="button"
            aria-label="Analyze"
            disabled={!canSend}
            onClick={onSubmit}
            className={`flex size-7 shrink-0 items-center justify-center transition-[background-color,color,transform] duration-200 enabled:active:scale-[0.94] ${
              pill ? "rounded-full" : "rounded-[8px]"
            } ${expanded ? "col-start-3 row-start-2" : "col-start-4 row-start-1"}`}
            style={{
              background: canSend ? "var(--ink)" : "var(--line-strong)",
              color: canSend ? "var(--surface)" : "var(--ink-2)",
            }}
          >
            <Icon size={16} strokeWidth={2.4}>
              <path d="M12 19V5M5 12l7-7 7 7" />
            </Icon>
          </button>
        </div>
      </div>
    </div>
  );
}
