/**
 * Insight Dashboard — post-call intelligence in the platform design language:
 * light page, white cards, uppercase section labels, dot-bars for intents
 * (matching the live session's INTENT DETECTION style), emotion trajectory,
 * eval scorecard, label gaps, and a call log with per-call drill-down.
 */

import React, { useCallback, useEffect, useState } from 'react';
import type {
  CountedLabel,
  EvalScorecard,
  InsightListResponse,
  InsightRecord,
  InsightSummary,
} from './types';
import './styles.css';

const POLL_MS = 10_000;

function getBackendUrl(): string {
  const w = window as unknown as { __BACKEND_URL__?: string };
  return w.__BACKEND_URL__ || 'http://localhost:7860';
}

/** snake_case intent -> display label */
const INTENT_LABELS: Record<string, string> = {
  dosing_administration: 'Dosing & Administration',
  safety_side_effects: 'Safety & Side Effects',
  adverse_event_report: 'Adverse Event Report',
  off_label_use: 'Off-label Use',
  drug_interactions: 'Drug Interactions',
  efficacy_evidence: 'Efficacy & Evidence',
  eligibility_trials: 'Trials & Eligibility',
  access_cost_insurance: 'Access, Cost & Insurance',
  device_usage_training: 'Device Usage & Training',
  competitor_comparison: 'Competitor Comparison',
  general_product_info: 'General Product Info',
  other: 'Other',
};

const EMOTION_EMOJI: Record<string, string> = {
  calm: '😌', anxious: '😟', frustrated: '😠', confused: '😕',
  urgent: '🚨', upbeat: '😊', reassured: '🤝',
};

function intentLabel(key: string): string {
  return INTENT_LABELS[key] || key.replace(/_/g, ' ');
}

/** Platform-style dot bar (like the session view's INTENT DETECTION ●●●●○○) */
function DotBar({ pct, tone }: { pct: number; tone?: 'ok' | 'warn' | 'bad' }) {
  const filled = Math.round((pct / 100) * 10);
  return (
    <span className={`ins-dots ins-dots-${tone || 'ok'}`}>
      {Array.from({ length: 10 }, (_, i) => (
        <span key={i} className={i < filled ? 'ins-dot-on' : 'ins-dot-off'} />
      ))}
    </span>
  );
}

function ProgressBar({ pct, color }: { pct: number; color: string }) {
  return (
    <div className="ins-progress">
      <div className="ins-progress-fill" style={{ width: `${Math.min(100, pct)}%`, background: color }} />
    </div>
  );
}

/** Parse eval scores like "27/27", "12/12", "4.93", "5" into a 0-100 pct */
function scoreToPct(score: string): number | null {
  const frac = score.match(/^(\d+(?:\.\d+)?)\s*\/\s*(\d+(?:\.\d+)?)$/);
  if (frac) return (100 * parseFloat(frac[1])) / parseFloat(frac[2]);
  const num = score.match(/^(\d+(?:\.\d+)?)$/);
  if (num) {
    const v = parseFloat(num[1]);
    if (v <= 5) return (100 * v) / 5; // 1-5 rubric scales
  }
  return null;
}

function Section({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <section className="ins-card">
      <h3 className="ins-label">{title}{hint && <span className="ins-hint">{hint}</span>}</h3>
      {children}
    </section>
  );
}

function IntentPanel({ intents }: { intents: CountedLabel[] }) {
  if (!intents.length) return <div className="ins-empty">no calls yet</div>;
  const total = intents.reduce((a, i) => a + i.count, 0) || 1;
  return (
    <div className="ins-rows">
      {intents.map((i) => {
        const pct = Math.round((100 * i.count) / total);
        const tone = i.label === 'adverse_event_report' ? 'bad'
          : i.label === 'off_label_use' || i.label === 'competitor_comparison' ? 'warn' : 'ok';
        return (
          <div key={i.label} className="ins-intent-row">
            <span className="ins-intent-name">{intentLabel(i.label)}</span>
            <DotBar pct={pct} tone={tone} />
            <span className="ins-intent-pct">{pct}%</span>
          </div>
        );
      })}
    </div>
  );
}

function EmotionPanel({ summary }: { summary: InsightSummary }) {
  const emotions = ['calm', 'upbeat', 'reassured', 'anxious', 'confused', 'frustrated', 'urgent'];
  const total = Object.values(summary.emotions_end).reduce((a, b) => a + b, 0) || 1;
  return (
    <>
      <div className="ins-rows">
        {emotions.map((e) => {
          const start = summary.emotions_start[e] || 0;
          const end = summary.emotions_end[e] || 0;
          if (!start && !end) return null;
          const negative = ['anxious', 'frustrated', 'confused', 'urgent'].includes(e);
          return (
            <div key={e} className="ins-emotion-row">
              <span className="ins-emotion-name">{EMOTION_EMOJI[e]} {e}</span>
              <ProgressBar pct={(100 * end) / total} color={negative ? '#dc2626' : '#16a34a'} />
              <span className="ins-emotion-delta">
                {start} → <b>{end}</b>
              </span>
            </div>
          );
        })}
      </div>
      <p className="ins-footnote">start → end of call · {summary.kpis.deescalated_pct}% of callers de-escalated</p>
    </>
  );
}

function EvalPanel({ evals }: { evals: EvalScorecard | null }) {
  if (!evals || !evals.baseline.available) {
    return <div className="ins-empty">run the eval harness to populate this panel</div>;
  }
  const advCompliance = evals.adversarial.metrics.find((m) => m.metric.includes('compliance'));
  return (
    <>
      <div className="ins-rows">
        {evals.baseline.metrics.map((m) => {
          const pct = scoreToPct(m.score);
          return (
            <div key={m.metric} className="ins-eval-row">
              <span className="ins-eval-name">{m.metric}</span>
              {pct !== null && <ProgressBar pct={pct} color={pct >= 90 ? '#16a34a' : pct >= 70 ? '#d97706' : '#dc2626'} />}
              <span className="ins-eval-score">{m.score}</span>
            </div>
          );
        })}
      </div>
      <p className="ins-footnote">
        adversarial red-team: <b>{advCompliance ? advCompliance.score : 'n/a'}</b> compliance
        {evals.baseline.failures > 0 && <> · {evals.baseline.failures} open failure{evals.baseline.failures > 1 ? 's' : ''}</>}
      </p>
    </>
  );
}

function BarList({ items, accent }: { items: CountedLabel[]; accent: string }) {
  if (!items.length) return <div className="ins-empty">none yet</div>;
  const max = Math.max(...items.map((i) => i.count));
  return (
    <div className="ins-rows">
      {items.slice(0, 8).map((i) => (
        <div key={i.label} className="ins-bar-row" title={i.label}>
          <div className="ins-bar-track">
            <div className="ins-bar-fill" style={{ width: `${(100 * i.count) / max}%`, background: accent }} />
            <span className="ins-bar-label">{i.label}</span>
          </div>
          <span className="ins-bar-count">{i.count}×</span>
        </div>
      ))}
    </div>
  );
}

function ChipList({ items }: { items: CountedLabel[] }) {
  if (!items.length) return <div className="ins-empty">none yet</div>;
  return (
    <div className="ins-chips">
      {items.map((i) => (
        <span key={i.label} className="ins-chip">{i.label} <b>{i.count}</b></span>
      ))}
    </div>
  );
}

function DrillDown({ record, onClose }: { record: InsightRecord; onClose: () => void }) {
  return (
    <div className="ins-drill-overlay" onClick={onClose}>
      <div className="ins-drill" onClick={(e) => e.stopPropagation()}>
        <div className="ins-drill-head">
          <div>
            <span className="ins-drill-id">call {record.call_id}</span>
            <span className="ins-drill-meta">
              {record.persona_id || 'unknown'} · {record.caller_type || '?'}
              {record.adverse_event_flag && <span className="ins-ae-badge"> ⚠ AE</span>}
              {record.resolved ? <span className="ins-ok"> · resolved</span> : <span className="ins-warn"> · unresolved</span>}
            </span>
          </div>
          <button className="ins-close" onClick={onClose}>✕</button>
        </div>
        <div className="ins-drill-body">
          <div className="ins-drill-col">
            <h4 className="ins-label">Transcript</h4>
            <pre className="ins-transcript">{record.transcript || '(not stored for this call)'}</pre>
          </div>
          <div className="ins-drill-col">
            <h4 className="ins-label">Extracted insight</h4>
            <div className="ins-kv"><span>Primary intent</span>
              <div className="ins-intent-chip">{intentLabel(record.primary_intent || 'other')}</div></div>
            {(record.secondary_intents || []).length > 0 && (
              <div className="ins-kv"><span>Also discussed</span>
                <div>{(record.secondary_intents || []).map(intentLabel).join(', ')}</div></div>
            )}
            <div className="ins-kv"><span>Emotion trajectory</span>
              <div>{EMOTION_EMOJI[record.emotion_start || ''] || ''} {record.emotion_start || '?'} →{' '}
                {EMOTION_EMOJI[record.emotion_end || ''] || ''} <b>{record.emotion_end || '?'}</b></div></div>
            <div className="ins-kv"><span>Unanswered (label gaps)</span>
              <div>{(record.unanswered || []).length
                ? (record.unanswered || []).map((u, idx) => (
                    <div key={idx} className="ins-unanswered">“{u.question}” <em>({u.reason})</em></div>
                  ))
                : '—'}</div></div>
            <div className="ins-kv"><span>Competitors</span>
              <div>{(record.competitor_mentions || []).join(', ') || '—'}</div></div>
            <div className="ins-kv"><span>Access barriers</span>
              <div>{(record.access_barriers || []).join(', ') || '—'}</div></div>
            <div className="ins-kv"><span>Captured</span><div>{record.captured_at}</div></div>
          </div>
        </div>
      </div>
    </div>
  );
}

export function InsightsDashboard({ onClose }: { onClose: () => void }) {
  const [summary, setSummary] = useState<InsightSummary | null>(null);
  const [records, setRecords] = useState<InsightRecord[]>([]);
  const [evals, setEvals] = useState<EvalScorecard | null>(null);
  const [error, setError] = useState<string>('');
  const [selected, setSelected] = useState<InsightRecord | null>(null);

  const refresh = useCallback(async () => {
    try {
      const base = getBackendUrl();
      const [sumRes, listRes, evalRes] = await Promise.all([
        fetch(`${base}/insights/summary`),
        fetch(`${base}/insights?limit=50`),
        fetch(`${base}/insights/evals`),
      ]);
      if (!sumRes.ok || !listRes.ok) throw new Error(`HTTP ${sumRes.status}/${listRes.status}`);
      setSummary((await sumRes.json()) as InsightSummary);
      setRecords(((await listRes.json()) as InsightListResponse).insights);
      if (evalRes.ok) setEvals((await evalRes.json()) as EvalScorecard);
      setError('');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'failed to load insights');
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, POLL_MS);
    return () => clearInterval(t);
  }, [refresh]);

  const k = summary?.kpis;

  return (
    <div className="ins-root">
      <div className="ins-header">
        <div className="ins-header-left">
          <span className="ins-header-orb" />
          <div>
            <h2>Insight Dashboard</h2>
            <span className="ins-sub">post-call intelligence · every call becomes data</span>
          </div>
        </div>
        <div className="ins-header-right">
          <span className="ins-live"><span className="ins-live-dot" /> Live · 10s</span>
          <button className="ins-close" onClick={onClose}>✕</button>
        </div>
      </div>

      {error && <div className="ins-error">⚠ {error} — is the backend running?</div>}

      <div className="ins-kpis">
        <div className="ins-kpi"><b>{k?.total_calls ?? '–'}</b><span>calls analyzed</span></div>
        <div className="ins-kpi ins-kpi-ae"><b>{k?.ae_flags ?? '–'}</b><span>AE reports</span></div>
        <div className="ins-kpi"><b>{k?.unanswered_questions ?? '–'}</b><span>label gaps found</span></div>
        <div className="ins-kpi ins-kpi-ok"><b>{k ? `${k.resolved_pct}%` : '–'}</b><span>calls resolved</span></div>
        <div className="ins-kpi ins-kpi-ok"><b>{k ? `${k.deescalated_pct}%` : '–'}</b><span>callers de-escalated</span></div>
      </div>

      <div className="ins-grid">
        <Section title="Intent Detection" hint="what callers actually want — medical intent taxonomy">
          <IntentPanel intents={summary?.intents ?? []} />
        </Section>
        <Section title="Emotion Detection" hint="caller state, start → end">
          {summary ? <EmotionPanel summary={summary} /> : <div className="ins-empty">…</div>}
        </Section>
        <Section title="Top Label Gaps" hint="questions the label couldn't answer — the content roadmap">
          <BarList items={summary?.label_gaps ?? []} accent="#dc2626" />
        </Section>
        <Section title="Eval Scorecard" hint="judged quality — SynthioLabs rubric + adversarial red-team">
          <EvalPanel evals={evals} />
        </Section>
      </div>

      <div className="ins-grid ins-grid-2">
        <Section title="Competitor Mentions">
          <ChipList items={summary?.competitor_mentions ?? []} />
        </Section>
        <Section title="Access / Cost Barriers">
          <ChipList items={summary?.access_barriers ?? []} />
        </Section>
      </div>

      <Section title="Call Log" hint="click a call for transcript + extracted insight">
        {!records.length && <div className="ins-empty">no calls captured yet — finish a voice session and it appears here</div>}
        <div className="ins-table">
          {records.map((r) => (
            <button key={`${r.call_id}-${r.captured_at}`} className="ins-row" onClick={() => setSelected(r)}>
              <span className="ins-row-id">#{r.call_id}</span>
              <span className="ins-row-persona">{r.persona_id || 'unknown'}</span>
              <span className="ins-intent-chip">{intentLabel(r.primary_intent || 'other')}</span>
              <span className="ins-row-emotion">
                {EMOTION_EMOJI[r.emotion_start || ''] || '·'}→{EMOTION_EMOJI[r.emotion_end || ''] || '·'}
              </span>
              <span className={r.resolved ? 'ins-ok' : 'ins-warn'}>{r.resolved ? 'resolved' : 'open'}</span>
              <span className={r.adverse_event_flag ? 'ins-ae-badge' : 'ins-muted'}>
                {r.adverse_event_flag ? '⚠ AE' : '—'}
              </span>
              <span className="ins-row-time">{(r.captured_at || '').replace('T', ' ').replace('Z', '')}</span>
            </button>
          ))}
        </div>
      </Section>

      {selected && <DrillDown record={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
