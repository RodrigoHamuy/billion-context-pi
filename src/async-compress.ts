import { createHash, randomUUID } from "node:crypto";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { CompressionBlock, CompressionCore, CompressionState, Config, CoreMessage } from "acp-kernel";
import { defaultCountTokens } from "acp-kernel";
import { normalizeRanges, tier3OnlyRewrite, blockSpanLabel } from "./compress-tool.js";
import { sanitizeSummary } from "./summary-sanitize.js";
import { adjustedTokenCount, collectCoveredMessageIds, collectImageTokens, estimateTokens, modelSupportsImages } from "./tokens.js";
import { getSystemPromptText } from "./compat.js";
import { logInfo, logWarn } from "./log.js";
import { ASYNC_CALL_ID_PREFIX, ASYNC_COMPRESS_CUSTOM_TYPE } from "./messages.js";

export { ASYNC_CALL_ID_PREFIX, ASYNC_COMPRESS_CUSTOM_TYPE };

export const ASYNC_FORK_TIMEOUT_MS = 5 * 60_000;
export const ASYNC_SUPPORTED_APIS: ReadonlySet<string> = new Set(["anthropic-messages", "openai-completions", "openai-responses", "openai-codex-responses"]);
const RESPONSES_APIS: ReadonlySet<string> = new Set(["openai-responses", "openai-codex-responses"]);

const SERVER_STATE_KEYS = ["previous_response_id", "conversation", "context_management"];

export interface AsyncRange {
  startRef: string;
  endRef: string;
  summary: string;
  topic?: string;
  summaryMaxChars?: number;
}

export interface AsyncCompressRecord {
  version: 1;
  callId: string;
  ranges: AsyncRange[];
  text: string;
}

export interface ViewSnapshot {
  count: number;
  digest: string;
  byRef: Record<string, string>;
  blocks: string[];
}

type Phase = "awaiting-request" | "awaiting-response" | "running" | "ready";

interface Job {
  id: string;
  sid: string;
  phase: Phase;
  nudgeText: string;
  snapshot: ViewSnapshot;
  api: string;
  provider: string;
  model: NonNullable<ExtensionContext["model"]>;
  thinkingLevel: ReturnType<AsyncCompressDeps["pi"]["getThinkingLevel"]>;
  tools: StreamContext["tools"];
  headers?: HeadersLike;
  payload?: unknown;
  controller: AbortController;
  timer?: ReturnType<typeof setTimeout>;
  ranges?: AsyncRange[];
}

type ProviderLike = NonNullable<ReturnType<ExtensionContext["modelRegistry"]["getProvider"]>>;
type StreamOptions = NonNullable<Parameters<ProviderLike["streamSimple"]>[2]>;
type StreamContext = Parameters<ProviderLike["streamSimple"]>[1];
type HeadersLike = NonNullable<StreamOptions["headers"]>;

function messageDigest(messages: readonly CoreMessage[], count: number): string {
  const hash = createHash("sha256");
  for (let i = 0; i < count; i++) {
    const m = messages[i]!;
    hash.update(`${m.id}\u0000${m.role}\u0000${m.contentType}\u0000${m.toolCallId ?? ""}\u0000${m.text ?? ""}\u0001`);
  }
  return hash.digest("hex");
}

function activeBlockKeys(state: CompressionState): string[] {
  return state.blocks.filter((b) => b.active).map((b) => `${b.blockId}:${b.runId}`).sort();
}

export function takeSnapshot(view: readonly CoreMessage[], state: CompressionState): ViewSnapshot {
  return {
    count: view.length,
    digest: messageDigest(view, view.length),
    byRef: { ...state.messageRefs.byRef },
    blocks: activeBlockKeys(state),
  };
}

export function snapshotStaleReason(snapshot: ViewSnapshot, view: readonly CoreMessage[], state: CompressionState, ranges: readonly AsyncRange[]): string | null {
  if (view.length < snapshot.count) return "view-shrank";
  if (messageDigest(view, snapshot.count) !== snapshot.digest) return "view-prefix-changed";
  const blocks = activeBlockKeys(state);
  if (blocks.length !== snapshot.blocks.length || blocks.some((b, i) => b !== snapshot.blocks[i])) return "blocks-changed";
  for (const r of ranges) {
    for (const ref of [r.startRef, r.endRef]) {
      if (!/^m\d+$/.test(ref)) continue;
      const then = snapshot.byRef[ref];
      if (then === undefined || state.messageRefs.byRef[ref] !== then) return `ref-rebound:${ref}`;
    }
  }
  return null;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

export function unsupportedPayloadReason(api: string, payload: unknown): string | null {
  if (!ASYNC_SUPPORTED_APIS.has(api)) return `unsupported-api:${api}`;
  if (!isRecord(payload)) return "payload-not-object";
  for (const key of SERVER_STATE_KEYS) if (key in payload) return `server-state:${key}`;
  const list = RESPONSES_APIS.has(api) ? payload.input : payload.messages;
  if (!Array.isArray(list) || list.length === 0) return "payload-shape";
  return null;
}

export function forkPayload(api: string, payload: unknown, nudgeText: string): unknown {
  const reason = unsupportedPayloadReason(api, payload);
  if (reason !== null || !isRecord(payload)) throw new Error(reason ?? "payload-not-object");
  const body = structuredClone(payload);
  if (RESPONSES_APIS.has(api)) {
    body.input = [...(body.input as unknown[]), { role: "user", content: [{ type: "input_text", text: nudgeText }] }];
  } else {
    body.messages = [...(body.messages as unknown[]), { role: "user", content: [{ type: "text", text: nudgeText }] }];
  }
  return body;
}

export function rangesFromToolArgs(args: unknown): AsyncRange[] | string {
  if (!isRecord(args)) return "compress arguments are not an object";
  const content = args.content;
  if (typeof content !== "string" && !Array.isArray(content)) return "compress arguments have no content";
  const topic = typeof args.topic === "string" ? args.topic : undefined;
  const summaryMaxChars = typeof args.summaryMaxChars === "number" ? args.summaryMaxChars : undefined;
  const parsed = normalizeRanges({ content: typeof content === "string" ? content : JSON.stringify(content), topic, summaryMaxChars });
  if (typeof parsed === "string") return parsed;
  if (parsed.length === 0) return "no ranges";
  return parsed.map((r) => ({
    startRef: r.startId,
    endRef: r.endId,
    summary: sanitizeSummary(r.summary).text,
    ...(r.topic ?? topic ? { topic: r.topic ?? topic } : {}),
    ...(summaryMaxChars !== undefined ? { summaryMaxChars } : {}),
  }));
}

export type AsyncApplyOutcome =
  | { ok: true; state: CompressionState; newBlocks: CompressionBlock[] }
  | { ok: false; kind: "stale" | "invalid" | "refold"; reason: string };

export function applyAsyncRanges(input: {
  core: CompressionCore;
  view: CoreMessage[];
  state: CompressionState;
  config: Config;
  tokenCount: number;
  snapshot: ViewSnapshot;
  ranges: AsyncRange[];
  callId: string;
}): AsyncApplyOutcome {
  const stale = snapshotStaleReason(input.snapshot, input.view, input.state, input.ranges);
  if (stale !== null) return { ok: false, kind: "stale", reason: stale };
  const probe = input.core.processTurn({ messages: input.view, state: structuredClone(input.state), config: input.config, tokenCount: input.tokenCount });
  const beforeRunIds = new Map(probe.state.blocks.map((b) => [b.blockId, b.runId]));
  const applied = input.core.applyCompression({
    ranges: input.ranges.map((r) => ({ ...r, compressCallId: input.callId })),
    messages: probe.messages,
    state: probe.state,
    config: input.config,
  });
  if (applied.result.errors.length > 0) return { ok: false, kind: "invalid", reason: applied.result.errors.slice(0, 3).join("; ") };
  if (applied.result.blocksCreated === 0) return { ok: false, kind: "invalid", reason: "no blocks created" };
  const newBlocks = applied.state.blocks.filter((b) => beforeRunIds.get(b.blockId) !== b.runId);
  // Kernel in-place refolds keep the block's original compressCallId, so the
  // async record could never be matched to the block again.
  const refolded = newBlocks.filter((b) => beforeRunIds.has(b.blockId));
  if (refolded.length > 0) return { ok: false, kind: "refold", reason: `in-place refold of ${refolded.map((b) => b.blockId).join(", ")}` };
  const rewrite = tier3OnlyRewrite(newBlocks, applied.state.blocks);
  if (rewrite) return { ok: false, kind: "invalid", reason: `tier-3-only rewrite ${rewrite.join(", ")}` };
  return { ok: true, state: applied.state, newBlocks };
}

export function isAsyncBlock(block: CompressionBlock): boolean {
  return block.compressCallId?.startsWith(ASYNC_CALL_ID_PREFIX) === true;
}

export function asyncCarriers(state: CompressionState): Map<string, { label: string; timestamp: number }> {
  const out = new Map<string, { label: string; timestamp: number }>();
  for (const b of state.blocks) {
    if (b.active && isAsyncBlock(b)) out.set(b.blockId, { label: blockSpanLabel(b, state), timestamp: b.createdAt });
  }
  return out;
}

export function asyncEnabledValue(value: unknown, warn: (v: unknown) => void): boolean {
  if (value === undefined || value === false) return false;
  if (value === true) return true;
  warn(value);
  return false;
}

export interface AsyncCompressDeps {
  pi: Pick<ExtensionAPI, "appendEntry" | "getThinkingLevel" | "getActiveTools" | "getAllTools">;
  now?: () => number;
  timeoutMs?: number;
}

export class AsyncCompressor {
  private readonly jobs = new Map<string, Job>();
  private readonly syncRetry = new Set<string>();
  private readonly fallback = new Map<string, string>();
  private readonly notified = new Set<string>();
  private readonly timeoutMs: number;

  constructor(private readonly deps: AsyncCompressDeps) {
    this.timeoutMs = deps.timeoutMs ?? ASYNC_FORK_TIMEOUT_MS;
  }

  isActive(sid: string): boolean {
    return this.jobs.has(sid);
  }

  fallbackReason(sid: string): string | undefined {
    return this.fallback.get(sid);
  }

  requestSyncRetry(sid: string): void {
    this.syncRetry.add(sid);
  }

  takeSyncRetry(sid: string): boolean {
    return this.syncRetry.delete(sid);
  }

  phase(sid: string): Phase | undefined {
    return this.jobs.get(sid)?.phase;
  }

  start(sid: string, init: { nudgeText: string; snapshot: ViewSnapshot; model: NonNullable<ExtensionContext["model"]> }): string {
    this.cancel(sid, "superseded");
    const id = randomUUID();
    this.jobs.set(sid, {
      id,
      sid,
      phase: "awaiting-request",
      nudgeText: init.nudgeText,
      snapshot: init.snapshot,
      api: String(init.model.api),
      provider: String(init.model.provider),
      model: { ...init.model },
      thinkingLevel: this.deps.pi.getThinkingLevel(),
      tools: this.activeTools(),
      controller: new AbortController(),
    });
    logInfo("async-compress", { sid, event: "job-created", job: id, api: init.model.api, view: init.snapshot.count });
    return id;
  }

  onHeaders(sid: string, headers: HeadersLike): void {
    const job = this.jobs.get(sid);
    if (job && job.phase === "awaiting-request" && job.headers === undefined) job.headers = headers;
  }

  onPayload(sid: string, payload: unknown, ctx: ExtensionContext): void {
    const job = this.jobs.get(sid);
    if (!job || job.phase !== "awaiting-request") return;
    const reason = unsupportedPayloadReason(job.api, payload);
    if (reason !== null) {
      this.abandon(job, `capture:${reason}`, true);
      this.markFallback(sid, reason, ctx);
      return;
    }
    job.payload = payload;
    job.phase = "awaiting-response";
  }

  onResponse(sid: string, status: number, ctx: ExtensionContext): void {
    const job = this.jobs.get(sid);
    if (!job || job.phase !== "awaiting-response") return;
    if (status >= 400) {
      this.abandon(job, `main-request-status:${status}`, true);
      return;
    }
    this.launch(job, ctx);
  }

  onStreamStart(sid: string, ctx: ExtensionContext): void {
    const job = this.jobs.get(sid);
    if (job && job.phase === "awaiting-response" && job.api === "openai-codex-responses") this.launch(job, ctx);
  }

  private launch(job: Job, ctx: ExtensionContext): void {
    job.phase = "running";
    void this.run(job, ctx);
  }

  onMainFailed(sid: string, reason: string, aborted = false): void {
    const job = this.jobs.get(sid);
    if (!job) return;
    if (job.phase === "awaiting-request" || job.phase === "awaiting-response") {
      this.abandon(job, reason, true);
    } else if (aborted && job.phase === "running") {
      this.abandon(job, reason, false);
    }
  }

  cancel(sid: string, reason: string): void {
    const job = this.jobs.get(sid);
    if (!job) return;
    this.abandon(job, reason, false);
  }

  resetSession(sid: string): void {
    this.cancel(sid, "session-reset");
    this.syncRetry.delete(sid);
    this.fallback.delete(sid);
    this.notified.delete(sid);
  }

  takeReady(sid: string): { id: string; ranges: AsyncRange[]; snapshot: ViewSnapshot } | undefined {
    const job = this.jobs.get(sid);
    if (!job || job.phase !== "ready" || !job.ranges) return undefined;
    this.jobs.delete(sid);
    return { id: job.id, ranges: job.ranges, snapshot: job.snapshot };
  }

  markFallback(sid: string, reason: string, ctx: ExtensionContext | undefined): void {
    if (!this.fallback.has(sid)) this.fallback.set(sid, reason);
    logWarn("async-compress", { sid, event: "sync-fallback", reason });
    if (!this.notified.has(sid)) {
      this.notified.add(sid);
      if (ctx?.hasUI) ctx.ui.notify(`[ACP] async compression unavailable (${reason}) — falling back to synchronous nudges for this session.`, "warning");
    }
  }

  private abandon(job: Job, reason: string, retrySync: boolean): void {
    if (this.jobs.get(job.sid) === job) this.jobs.delete(job.sid);
    if (job.timer) clearTimeout(job.timer);
    job.controller.abort();
    if (retrySync) this.syncRetry.add(job.sid);
    logInfo("async-compress", { sid: job.sid, event: "job-dropped", job: job.id, phase: job.phase, reason, syncRetry: retrySync });
  }

  private async run(job: Job, ctx: ExtensionContext): Promise<void> {
    const started = (this.deps.now ?? Date.now)();
    let timedOut = false;
    const deadline = new Promise<"deadline">((resolve) => {
      job.timer = setTimeout(() => {
        timedOut = true;
        job.controller.abort();
        resolve("deadline");
      }, this.timeoutMs);
    });
    const fork = this.fork(job, ctx);
    fork.catch(() => {});
    try {
      // Settles even when the provider or auth lookup ignores the abort signal.
      const message = await Promise.race([fork, deadline]);
      if (this.jobs.get(job.sid) !== job) return;
      if (message === "deadline") {
        this.abandon(job, "fork-timeout", false);
        this.markFallback(job.sid, "fork-timeout", ctx);
        return;
      }
      const usage = message.usage;
      logInfo("async-compress", {
        sid: job.sid,
        event: "fork-finished",
        job: job.id,
        stopReason: message.stopReason,
        ms: (this.deps.now ?? Date.now)() - started,
        input: usage?.input ?? null,
        output: usage?.output ?? null,
        cacheRead: usage?.cacheRead ?? null,
        cacheWrite: usage?.cacheWrite ?? null,
      });
      if (message.stopReason === "error" || message.stopReason === "aborted") {
        this.abandon(job, `fork-${message.stopReason}`, false);
        this.markFallback(job.sid, timedOut ? "fork-timeout" : "fork-error", ctx);
        return;
      }
      const call = message.content.find((c) => c.type === "toolCall" && c.name === "compress");
      if (!call || call.type !== "toolCall") {
        this.abandon(job, "fork-no-compress-call", false);
        return;
      }
      const ranges = rangesFromToolArgs(call.arguments);
      if (typeof ranges === "string") {
        this.abandon(job, "fork-invalid-args", false);
        this.markFallback(job.sid, "invalid-compress-args", ctx);
        return;
      }
      job.ranges = ranges;
      job.phase = "ready";
      logInfo("async-compress", { sid: job.sid, event: "result-ready", job: job.id, ranges: ranges.length });
    } catch (e) {
      if (this.jobs.get(job.sid) !== job) return;
      // Provider error text can echo credentials: log the error class only.
      logWarn("async-compress", { sid: job.sid, event: "fork-threw", job: job.id, errorKind: e instanceof Error ? e.name : typeof e });
      this.abandon(job, "fork-threw", false);
      this.markFallback(job.sid, "fork-error", ctx);
    } finally {
      if (job.timer) clearTimeout(job.timer);
    }
  }

  private async fork(job: Job, ctx: ExtensionContext) {
    const provider = ctx.modelRegistry.getProvider(job.provider);
    if (!provider) throw new Error("unknown provider");
    const auth = await ctx.modelRegistry.getProviderAuth(job.provider);
    if (job.controller.signal.aborted) throw new Error("cancelled before transport");
    const payload = forkPayload(job.api, job.payload, job.nudgeText);
    const options: StreamOptions = {
      apiKey: auth?.auth.apiKey,
      env: auth?.env,
      headers: job.headers ? structuredClone(job.headers) : undefined,
      sessionId: job.sid,
      signal: job.controller.signal,
      onPayload: () => payload,
      // Codex WebSockets are cached per session id with continuation state; the fork must not share them.
      ...(job.api === "openai-codex-responses" ? { transport: "sse" as const } : {}),
      ...(job.thinkingLevel !== "off" ? { reasoning: job.thinkingLevel } : {}),
    };
    const requestModel = auth?.auth.baseUrl ? { ...job.model, baseUrl: auth.auth.baseUrl } : job.model;
    const context: StreamContext = { systemPrompt: "", messages: [], tools: job.tools };
    return provider.streamSimple(requestModel, context, options).result();
  }

  private activeTools(): StreamContext["tools"] {
    const active = new Set(this.deps.pi.getActiveTools());
    return this.deps.pi.getAllTools()
      .filter((t) => active.has(t.name))
      .map((t) => ({ name: t.name, description: t.description, parameters: t.parameters }));
  }
}

/** Caller holds the session lock. Order: validate → append replay record
 *  (failure discards the result) → save sidecar → adopt state. */
export async function applyReadyAsyncResult(input: {
  compressor: AsyncCompressor;
  pi: Pick<ExtensionAPI, "appendEntry">;
  ctx: ExtensionContext;
  core: CompressionCore;
  config: Config;
  view: CoreMessage[];
  state: CompressionState;
  entries: Parameters<typeof collectImageTokens>[0];
  save: (state: CompressionState) => Promise<void>;
  enabled: boolean;
}): Promise<{ state: CompressionState; record: AsyncCompressRecord } | undefined> {
  const { compressor, ctx } = input;
  const sid = ctx.sessionManager.getSessionId();
  if (!input.enabled) {
    compressor.cancel(sid, "disabled");
    return undefined;
  }
  const ready = compressor.takeReady(sid);
  if (!ready) return undefined;
  const callId = `${ASYNC_CALL_ID_PREFIX}${ready.id}`;
  const systemPromptText = getSystemPromptText(ctx);
  const systemPromptTokens = systemPromptText ? defaultCountTokens(systemPromptText) : 0;
  const imageTokens = collectImageTokens(input.entries, modelSupportsImages(ctx.model));
  const prelim = estimateTokens(input.view, collectCoveredMessageIds(input.state), imageTokens) + systemPromptTokens;
  const tokenCount = adjustedTokenCount(input.core, input.view, input.state, input.config, prelim, imageTokens, systemPromptTokens);
  let outcome: AsyncApplyOutcome;
  try {
    outcome = applyAsyncRanges({ core: input.core, view: input.view, state: input.state, config: input.config, tokenCount, snapshot: ready.snapshot, ranges: ready.ranges, callId });
  } catch (e) {
    outcome = { ok: false, kind: "invalid", reason: e instanceof Error ? e.message : String(e) };
  }
  if (!outcome.ok) {
    logInfo("async-compress", { sid, event: "result-discarded", job: ready.id, kind: outcome.kind, reason: outcome.reason });
    if (outcome.kind === "invalid") compressor.markFallback(sid, "invalid-result", ctx);
    if (outcome.kind === "refold") compressor.requestSyncRetry(sid);
    return undefined;
  }
  const label = outcome.newBlocks.map((b) => blockSpanLabel(b, outcome.state)).join(", ");
  const record: AsyncCompressRecord = { version: 1, callId, ranges: ready.ranges, text: `▣ ACP async compress | blocks: ${label}` };
  try {
    if (typeof input.pi.appendEntry !== "function") throw new Error("appendEntry unavailable");
    input.pi.appendEntry(ASYNC_COMPRESS_CUSTOM_TYPE, record);
  } catch (e) {
    logWarn("async-compress", { sid, event: "record-append-failed", job: ready.id, error: e instanceof Error ? e.message : String(e) });
    compressor.markFallback(sid, "record-append-failed", ctx);
    return undefined;
  }
  await input.save(outcome.state);
  logInfo("async-compress", { sid, event: "applied", job: ready.id, callId, blocks: outcome.newBlocks.map((b) => b.blockId) });
  if (ctx.hasUI) ctx.ui.notify(`[ACP] ${record.text}`);
  return { state: outcome.state, record };
}
