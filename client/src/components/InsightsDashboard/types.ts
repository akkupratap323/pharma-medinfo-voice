/**
 * Types for the Insights Dashboard — mirrors the backend contract:
 *   GET /insights          -> { total, insights: InsightRecord[] }
 *   GET /insights/summary  -> InsightSummary
 *   GET /insights/evals    -> EvalScorecard
 * (see app/api/routes.py insight endpoints and app/services/insight_capture.py)
 */

export interface UnansweredQuestion {
  question: string;
  reason: 'off-label' | 'not-in-label' | 'escalated' | 'other' | string;
}

export type MedicalIntent =
  | 'dosing_administration'
  | 'safety_side_effects'
  | 'adverse_event_report'
  | 'off_label_use'
  | 'drug_interactions'
  | 'efficacy_evidence'
  | 'eligibility_trials'
  | 'access_cost_insurance'
  | 'device_usage_training'
  | 'competitor_comparison'
  | 'general_product_info'
  | 'other'
  | string;

export type Emotion =
  | 'calm'
  | 'anxious'
  | 'frustrated'
  | 'confused'
  | 'urgent'
  | 'upbeat'
  | 'reassured'
  | string;

export interface InsightRecord {
  call_id: string;
  persona_id: string;
  captured_at: string;
  transcript?: string;
  primary_intent?: MedicalIntent;
  secondary_intents?: MedicalIntent[];
  themes?: string[];
  unanswered?: UnansweredQuestion[];
  competitor_mentions?: string[];
  access_barriers?: string[];
  adverse_event_flag?: boolean;
  emotion_start?: Emotion;
  emotion_end?: Emotion;
  resolved?: boolean;
  sentiment?: 'positive' | 'neutral' | 'frustrated' | string;
  caller_type?: 'hcp' | 'patient' | 'field_rep' | 'unknown' | string;
}

export interface CountedLabel {
  label: string;
  count: number;
}

export interface InsightSummary {
  kpis: {
    total_calls: number;
    ae_flags: number;
    unanswered_questions: number;
    resolved_pct: number;
    deescalated_pct: number;
    positive_pct: number;
  };
  intents: CountedLabel[];
  emotions_start: Record<string, number>;
  emotions_end: Record<string, number>;
  caller_types: Record<string, number>;
  label_gaps: CountedLabel[];
  themes: CountedLabel[];
  competitor_mentions: CountedLabel[];
  access_barriers: CountedLabel[];
  sentiment: Record<string, number>;
  personas: Record<string, number>;
}

export interface InsightListResponse {
  total: number;
  insights: InsightRecord[];
}

export interface EvalMetric {
  metric: string;
  score: string;
}

export interface EvalReport {
  available: boolean;
  metrics: EvalMetric[];
  failures: number;
}

export interface EvalScorecard {
  baseline: EvalReport;
  adversarial: EvalReport;
}
