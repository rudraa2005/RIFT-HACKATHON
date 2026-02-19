export function buildStrictExportPayload(analysis) {
  const suspicious_accounts = (analysis?.suspicious_accounts || []).map((a) => ({
    account_id: a?.account_id ?? '',
    suspicion_score: Number.isFinite(Number(a?.suspicion_score)) ? Number(a.suspicion_score) : 0,
    detected_patterns: Array.isArray(a?.detected_patterns) ? a.detected_patterns : [],
    ring_id: a?.ring_id ?? 'RING_NONE',
  }))

  const fraud_rings = (analysis?.fraud_rings || []).map((r) => ({
    ring_id: r?.ring_id ?? '',
    member_accounts: Array.isArray(r?.member_accounts) ? r.member_accounts : [],
    pattern_type: r?.pattern_type ?? '',
    risk_score: Number.isFinite(Number(r?.risk_score)) ? Number(r.risk_score) : 0,
  }))

  const summary = {
    total_accounts_analyzed: Number.isFinite(Number(analysis?.summary?.total_accounts_analyzed))
      ? Number(analysis.summary.total_accounts_analyzed)
      : 0,
    suspicious_accounts_flagged: Number.isFinite(Number(analysis?.summary?.suspicious_accounts_flagged))
      ? Number(analysis.summary.suspicious_accounts_flagged)
      : suspicious_accounts.length,
    fraud_rings_detected: Number.isFinite(Number(analysis?.summary?.fraud_rings_detected))
      ? Number(analysis.summary.fraud_rings_detected)
      : fraud_rings.length,
    processing_time_seconds: Number.isFinite(Number(analysis?.summary?.processing_time_seconds))
      ? Number(analysis.summary.processing_time_seconds)
      : 0,
  }

  return {
    suspicious_accounts,
    fraud_rings,
    summary,
  }
}
