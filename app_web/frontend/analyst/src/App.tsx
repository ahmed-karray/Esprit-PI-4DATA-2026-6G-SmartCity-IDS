import { ChangeEvent, FormEvent, ReactNode, useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ROLE_CONFIG, SupportedRole } from './role';

type DatasetOption = { dataset: string; features: string[] };
type PredictionResponse = Record<string, unknown>;
type SummaryRow = Record<string, unknown>;
type TimelineRow = Record<string, unknown>;
type HistoryRow = Record<string, unknown>;
type AuthUser = {
  id: number;
  full_name: string;
  email: string;
  role: SupportedRole;
  status: string;
  is_active: boolean;
  last_login?: string;
};
type AuthResponse = {
  access_token: string;
  token_type: string;
  user: AuthUser;
};
type RegistrationResponse = {
  message: string;
  status: string;
};
type StatTileProps = {
  label: string;
  value: string;
  hint?: string;
};

const ANALYST_FALLBACK_DATASETS: DatasetOption[] = [
  {
    dataset: 'eMBB',
    features: ['Dur', 'TotPkts', 'TotBytes', 'Rate', 'Load', 'Loss', 'pLoss', 'TcpRtt'],
  },
  {
    dataset: 'mMTC',
    features: ['TotPkts', 'Rate', 'SrcGap', 'DstGap', 'Dur', 'Load', 'Loss', 'TcpRtt'],
  },
  {
    dataset: 'URLLC',
    features: ['TcpRtt', 'SynAck', 'AckDat', 'Loss', 'Dur', 'Rate', 'TotPkts', 'TotBytes'],
  },
  {
    dataset: 'TON_IoT',
    features: ['src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts', 'duration', 'proto', 'conn_state', 'service'],
  },
];

const QUICK_ACCOUNTS = [
  {
    label: 'Security Analyst',
    role: 'security_analyst' as SupportedRole,
    email: 'analyst@hexamind.local',
    password: 'analyst123',
    copy: 'Investigate live detections, batch uploads, and dashboard alerts.',
    tone: 'success' as const,
  },
  {
    label: 'Data Scientist',
    role: 'data_scientist' as SupportedRole,
    email: 'scientist@hexamind.local',
    password: 'scientist123',
    copy: 'Monitor drift, compare experiments, and promote champion models.',
    tone: 'info' as const,
  },
  {
    label: 'Administrator',
    role: 'administrator' as SupportedRole,
    email: 'admin@hexamind.local',
    password: 'admin123',
    copy: 'Manage users, platform health, and operational detection policy.',
    tone: 'warning' as const,
  },
] as const;

const API_BASE = 'http://localhost:8010';
const GATEWAY_LOGIN_URL = `${API_BASE}/login`;
const STORAGE_KEY = 'iotinel_user_session';
const IS_GATEWAY = window.location.port === '8010';
const APP_PREFIX = IS_GATEWAY ? '/analyst' : '';
const LOGIN_PATH = '/login';
const HOME_PATH = `${APP_PREFIX}${ROLE_CONFIG.homePath}`;
const FINAL_THRESHOLD = 0.3;

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    credentials: 'include',
    ...init,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = payload.detail;
    let message = payload.error || `Request failed: ${path}`;
    if (typeof detail === 'string') {
      message = detail;
    } else if (Array.isArray(detail)) {
      message = detail
        .map((entry) => String(entry?.msg || entry?.message || JSON.stringify(entry)))
        .join(', ');
    } else if (detail && typeof detail === 'object') {
      message = String((detail as { msg?: string }).msg || JSON.stringify(detail));
    }
    throw new Error(String(message));
  }
  return response.json() as Promise<T>;
}

function isAuthError(message: string): boolean {
  return ['Authentication required', 'Invalid token', 'User not available', 'Insufficient permissions'].some((token) =>
    message.includes(token),
  );
}

function readStoredUser(): AuthUser | null {
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as AuthUser;
  } catch {
    return null;
  }
}

function writeStoredUser(user: AuthUser | null): void {
  if (!user) {
    window.localStorage.removeItem(STORAGE_KEY);
    return;
  }
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(user));
}

function routeFor(path: string): string {
  return `${APP_PREFIX}${path}`;
}

function normalizePath(pathname: string): string {
  if (APP_PREFIX && pathname.startsWith(APP_PREFIX)) {
    const trimmed = pathname.slice(APP_PREFIX.length);
    return trimmed || '/';
  }
  return pathname || '/';
}

function redirectForRole(role: SupportedRole): string {
  return ROLE_CONFIG.redirectMap[role];
}

function toNumber(value: unknown, fallback = 0): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

function isMaliciousPredictionValue(value: unknown): boolean {
  if (typeof value === 'number') return value >= 1;
  if (typeof value === 'boolean') return value;
  const normalized = String(value || '')
    .trim()
    .toLowerCase();
  return ['1', 'malicious', 'attack', 'anomaly', 'intrusion', 'suspicious'].includes(normalized);
}

function classifyAttackType(datasetName: string, row: Record<string, unknown>): string {
  const g = (col: string, defaultValue: unknown = null) => row[col] ?? defaultValue;

  if (datasetName === 'eMBB') {
    const dur = toNumber(g('Dur', 1.0), 1.0);
    const rate = toNumber(g('Rate', 0.0), 0.0);
    const loss = toNumber(g('Loss', 0.0), 0.0);
    if (dur <= 0.35 && rate > 50) return 'TCP SYN Flood';
    if (loss > 0.04) return 'Bandwidth Saturation';
    return 'Bandwidth Saturation';
  }

  if (datasetName === 'mMTC') {
    const dur = toNumber(g('Dur', 1.0), 1.0);
    const totPkts = toNumber(g('TotPkts', 100), 100);
    const load = toNumber(g('Load', 5000), 5000);
    if (totPkts < 10 && dur < 1.0) return 'TCP SYN Scan / Connection Flooding';
    if (dur > 3.0 && load < 3000) return 'Slow-Rate Resource Exhaustion';
    return 'FIN Scan';
  }

  if (datasetName === 'URLLC') {
    const dur = toNumber(g('Dur', 1.0), 1.0);
    const totPkts = toNumber(g('TotPkts', 10), 10);
    const tcpRtt = toNumber(g('TcpRtt', 0.01), 0.01);
    if (dur === 0 && totPkts === 1) return 'UDP DDoS Flood';
    if (tcpRtt > 0.05 && totPkts < 5) return 'RST Injection';
    if (tcpRtt > 0.1) return 'SLA Violation / DoS';
    return 'Reconnaissance';
  }

  if (datasetName === 'TON_IoT') {
    const srcPkts = toNumber(g('src_pkts', 10), 10);
    const dstBytes = toNumber(g('dst_bytes', 1000), 1000);
    const srcBytes = toNumber(g('src_bytes', 1000), 1000);
    const duration = toNumber(g('duration', 1.0), 1.0);
    const connState = String(g('conn_state', 'SF'));
    const service = String(g('service', '-'));
    const proto = String(g('proto', 'tcp'));
    if (srcPkts > 1000 && duration < 2.0) return 'DDoS';
    if (srcPkts > 200 && duration < 5.0) return 'DoS';
    if (['REJ', 'RSTO', 'RSTOS0'].includes(connState) && srcPkts < 10) return 'Scanning';
    if (['ssh', 'ftp', 'ftp-data'].includes(service) && srcPkts < 20) return 'Password';
    if (duration > 60 && srcBytes < 5000 && connState === 'SF') return 'Backdoor';
    if (srcBytes > 100000 && duration > 10) return 'Ransomware';
    if (['http', 'http-alt'].includes(service)) {
      if (dstBytes > srcBytes * 2) return 'Injection';
      return 'XSS';
    }
    if (!['tcp', 'udp'].includes(proto) || (connState === 'SF' && srcBytes < 500)) return 'MITM';
    return 'DoS';
  }

  return 'Unknown';
}

function derivePredictionViewModel(
  datasetName: string,
  prediction: PredictionResponse | null,
  row: Record<string, unknown>,
): { label: string; attackType: string | null; confidence: number } {
  const confidence = toNumber(prediction?.confidence, 0);
  const binary = isMaliciousPredictionValue(prediction?.prediction);

  if (!binary) {
    return { label: 'Benign', attackType: null, confidence };
  }
  if (confidence < FINAL_THRESHOLD) {
    return { label: 'False Alarm', attackType: null, confidence };
  }

  const serverAttackType = prediction?.attack_type;
  const normalizedServerAttackType =
    typeof serverAttackType === 'string' ? serverAttackType.trim().toLowerCase() : '';
  const shouldUseRuleBasedAttackType =
    !normalizedServerAttackType || normalizedServerAttackType.startsWith('generic ');
  const attackType =
    !shouldUseRuleBasedAttackType && typeof serverAttackType === 'string' && serverAttackType.trim()
      ? serverAttackType.trim()
      : classifyAttackType(datasetName, row);

  return { label: 'Malicious', attackType, confidence };
}

async function getCohereExplanation(
  datasetName: string,
  attackType: string | null,
  label: string,
  confidence: number,
  row: Record<string, unknown>,
): Promise<string> {
  try {
    const payload = await api<{ explanation?: string }>('/detect/ai-explanation', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset: datasetName,
        attack_type: attackType,
        label,
        confidence,
        row,
      }),
    });
    return payload.explanation?.trim() || 'AI explanation unavailable';
  } catch {
    return 'AI explanation unavailable';
  }
}

function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randomFloat(min: number, max: number, digits = 2): string {
  return (Math.random() * (max - min) + min).toFixed(digits);
}

function randomChoice(values: string[]): string {
  return values[randomInt(0, values.length - 1)];
}

function randomValueForFeature(datasetName: string, feature: string): string {
  const lowerFeature = feature.toLowerCase();
  const lowerDataset = datasetName.toLowerCase();

  if (lowerFeature.includes('proto')) {
    return randomChoice(['tcp', 'udp', 'icmp']);
  }
  if (lowerFeature.includes('conn_state')) {
    return randomChoice(['REJ', 'RSTO', 'RSTR', 'S0', 'S1', 'SF', 'SH']);
  }
  if (lowerFeature.includes('service')) {
    return randomChoice(['http', 'dns', 'ftp', 'ssh', 'smb', 'dhcp']);
  }
  if (lowerFeature.includes('loss') || lowerFeature.includes('ploss')) {
    return randomFloat(0, 0.25, 3);
  }
  if (lowerFeature.includes('rate') || lowerFeature.includes('load')) {
    return randomFloat(0.1, 250, 2);
  }
  if (lowerFeature.includes('rtt') || lowerFeature.includes('gap') || lowerFeature.includes('ack')) {
    return randomFloat(1, 200, 2);
  }
  if (lowerFeature.includes('dur') || lowerFeature.includes('duration')) {
    return lowerDataset === 'ton_iot' ? String(randomInt(1, 30)) : randomFloat(1, 120, 2);
  }
  if (lowerFeature.includes('bytes')) {
    return String(randomInt(40, 9000));
  }
  if (lowerFeature.includes('pkts') || lowerFeature.includes('packets')) {
    return String(randomInt(1, 500));
  }
  if (lowerFeature.includes('syn') || lowerFeature.includes('tot')) {
    return String(randomInt(1, 1000));
  }
  return String(randomInt(1, 100));
}

function buildRandomFeatureValues(datasetName: string, featureList: string[]): Record<string, string> {
  return Object.fromEntries(featureList.map((feature) => [feature, randomValueForFeature(datasetName, feature)]));
}

function StatTile({ label, value, hint }: StatTileProps) {
  return (
    <div className="insight-tile">
      <div className="insight-label">{label}</div>
      <div className="insight-value">{value}</div>
      {hint ? <div className="insight-hint">{hint}</div> : null}
    </div>
  );
}

function SparkBars({ items, tone = 'cyan' }: { items: Array<{ label: string; value: number }>; tone?: 'cyan' | 'emerald' | 'amber' }) {
  const max = Math.max(...items.map((item) => item.value), 1);
  return (
    <div className="spark-bars">
      {items.map((item) => (
        <div key={item.label} className="spark-row">
          <div className="spark-meta">
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </div>
          <div className="spark-track">
            <div className={`spark-fill ${tone}`} style={{ width: `${(item.value / max) * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function MiniAreaChart({ points, title, xAxisLabel, yAxisLabel }: { points: number[]; title?: string; xAxisLabel?: string; yAxisLabel?: string }) {
  if (!points.length) {
    return <div className="empty-state">No chart data available yet.</div>;
  }

  // ✅ Fix: guard against single data point (renders broken triangle otherwise)
  if (points.length === 1) {
    return (
      <div style={{ width: '100%' }}>
        {title && (
          <div style={{ textAlign: 'center', fontSize: 14, fontWeight: 600, color: 'rgba(255,255,255,0.85)', marginBottom: 8 }}>
            {title}
          </div>
        )}
        <div style={{ textAlign: 'center', padding: '32px 16px', borderRadius: 12, background: 'rgba(255,255,255,0.03)', border: '1px dashed rgba(255,255,255,0.1)' }}>
          <div style={{ fontSize: 32, fontWeight: 700, color: '#22d3ee' }}>{points[0]}</div>
          <div style={{ marginTop: 8, fontSize: 13, color: 'rgba(255,255,255,0.4)' }}>
            Only 1 data point — run more predictions to build a trend line.
          </div>
        </div>
      </div>
    );
  }

  const width = 800;
  const height = 400;
  const padding = { top: 40, right: 60, bottom: 60, left: 80 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const max = Math.max(...points, 1);
  const min = Math.min(...points, 0);
  const range = max - min || 1;
  // ✅ Fix: safe step calculation, never divides by zero
  const step = chartWidth / (points.length - 1);

  const coords = points.map((value, i) => ({
    x: padding.left + i * step,
    y: padding.top + chartHeight - ((value - min) / range) * chartHeight,
  }));

  const polylinePoints = coords.map((c) => `${c.x},${c.y}`).join(' ');
  const areaPoints = [
    `${coords[0].x},${padding.top + chartHeight}`,   // ✅ Fix: start from first point's x, not always padding.left
    ...coords.map((c) => `${c.x},${c.y}`),
    `${coords[coords.length - 1].x},${padding.top + chartHeight}`, // ✅ Fix: end at last point's x
  ].join(' ');

  const yTicks = 5;
  const yTickValues = Array.from({ length: yTicks }, (_, i) => min + (range * i) / (yTicks - 1));
  const xTicks = Math.min(points.length, 6);
  const xTickIndices = Array.from({ length: xTicks }, (_, i) =>
    Math.round(((points.length - 1) * i) / Math.max(xTicks - 1, 1))
  );

  const gradientId = `area-gradient-${Math.random().toString(36).slice(2, 7)}`;

  return (
    <div style={{ width: '100%' }}>
      {title && (
        <div style={{ textAlign: 'center', fontSize: 14, fontWeight: 600, color: 'rgba(255,255,255,0.85)', marginBottom: 8 }}>
          {title}
        </div>
      )}
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ width: '100%', height: 'auto', display: 'block' }}
        preserveAspectRatio="xMidYMid meet"
      >
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.35" />
            <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.02" />
          </linearGradient>
        </defs>

        {yTickValues.map((value, i) => {
          const y = padding.top + chartHeight - ((value - min) / range) * chartHeight;
          return (
            <line key={`grid-${i}`} x1={padding.left} y1={y} x2={padding.left + chartWidth} y2={y}
              stroke="rgba(255,255,255,0.07)" strokeWidth="1" />
          );
        })}

        <polygon points={areaPoints} fill={`url(#${gradientId})`} />

        <polyline points={polylinePoints} fill="none" stroke="#22d3ee"
          strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />

        {coords.map((c, i) => (
          <circle key={`pt-${i}`} cx={c.x} cy={c.y} r="5" fill="#22d3ee" stroke="#0f172a" strokeWidth="2" />
        ))}

        <line x1={padding.left} y1={padding.top} x2={padding.left} y2={padding.top + chartHeight}
          stroke="rgba(255,255,255,0.2)" strokeWidth="2" />
        <line x1={padding.left} y1={padding.top + chartHeight} x2={padding.left + chartWidth} y2={padding.top + chartHeight}
          stroke="rgba(255,255,255,0.2)" strokeWidth="2" />

        {yTickValues.map((value, i) => {
          const y = padding.top + chartHeight - ((value - min) / range) * chartHeight;
          return (
            <text key={`yl-${i}`} x={padding.left - 10} y={y + 5}
              textAnchor="end" fontSize="13" fill="rgba(255,255,255,0.6)" fontWeight="500">
              {value >= 1000 ? `${(value / 1000).toFixed(1)}k` : value.toFixed(0)}
            </text>
          );
        })}

        {xTickIndices.map((idx, i) => (
          <text key={`xl-${i}`} x={padding.left + idx * step} y={padding.top + chartHeight + 24}
            textAnchor="middle" fontSize="13" fill="rgba(255,255,255,0.6)" fontWeight="500">
            {idx + 1}
          </text>
        ))}

        {yAxisLabel && (
          <text x={20} y={padding.top + chartHeight / 2} textAnchor="middle" fontSize="13"
            fill="rgba(255,255,255,0.7)" fontWeight="600"
            transform={`rotate(-90, 20, ${padding.top + chartHeight / 2})`}>
            {yAxisLabel}
          </text>
        )}

        {xAxisLabel && (
          <text x={padding.left + chartWidth / 2} y={height - 8}
            textAnchor="middle" fontSize="13" fill="rgba(255,255,255,0.7)" fontWeight="600">
            {xAxisLabel}
          </text>
        )}

        <text x={padding.left + chartWidth + 8} y={padding.top + 6}
          fontSize="12" fill="rgba(34,211,238,0.85)" fontWeight="700">
          Max: {max >= 1000 ? `${(max / 1000).toFixed(1)}k` : max.toFixed(0)}
        </text>
      </svg>
    </div>
  );
}

function Shell({
  user,
  title,
  subtitle,
  activePath,
  onLogout,
  children,
}: {
  user: AuthUser | null;
  title: string;
  subtitle: string;
  activePath: string;
  onLogout: () => void;
  children: ReactNode;
}) {
  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="brand-mark">A</div>
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.28em] text-cyan-300">IOTinel</div>
              <div className="text-lg font-semibold text-slate-50">Analyst Workspace</div>
            </div>
          </div>
          <div className="rounded-2xl border border-white/10 bg-slate-950/70 p-4">
            <div className="text-sm font-semibold">{user?.full_name || 'Security Analyst'}</div>
            <div className="mt-1 text-sm text-slate-400">{user?.email || 'signed in'}</div>
            <div className="mt-3 status-pill success">Role: Analyst</div>
          </div>
        </div>

        <div>
          <div className="sidebar-section-label">Operations</div>
          <nav className="space-y-2">
            {ROLE_CONFIG.menu.map((item) => (
              <a key={item.path} href={routeFor(item.path)} className={`nav-link ${activePath === item.path ? 'active' : ''}`}>
                <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-white/5 text-xs font-semibold">
                  {item.label.slice(0, 1)}
                </span>
                <div>
                  <div className="font-medium">{item.label}</div>
                  <div className="text-xs text-slate-500">Gateway-backed route</div>
                </div>
              </a>
            ))}
          </nav>
        </div>

        <div className="mt-auto">
          <button className="button-secondary w-full" onClick={onLogout}>
            Sign Out
          </button>
        </div>
      </aside>

      <div className="main-shell">
        <header className="topbar">
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.24em] text-cyan-300">Live 6G IDS</div>
            <h1 className="mt-2 text-3xl font-semibold">{title}</h1>
            <p className="mt-2 text-sm text-slate-400">{subtitle}</p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-slate-950/70 px-4 py-3 text-sm text-slate-300">
            <div className="font-semibold text-slate-50">Session active</div>
            <div className="mt-1">Cookie auth + gateway proxy</div>
          </div>
        </header>
        <main className="content-area">
          <div className="page-stack">{children}</div>
        </main>
      </div>
    </div>
  );
}

function SectionCard({ title, action, children }: { title: string; action?: ReactNode; children: ReactNode }) {
  return (
    <section className="surface-card p-6">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <h2 className="text-2xl font-semibold">{title}</h2>
        {action}
      </div>
      <div className="mt-5">{children}</div>
    </section>
  );
}

function LoginView({
  email,
  password,
  error,
  loading,
  registerError,
  registerLoading,
  registerSuccess,
  registerForm,
  onLoginChange,
  onQuickFill,
  onRegisterChange,
  onSubmit,
  onRegisterSubmit,
}: {
  email: string;
  password: string;
  error: string;
  loading: boolean;
  registerError: string;
  registerLoading: boolean;
  registerSuccess: string;
  registerForm: { fullName: string; email: string; password: string; role: SupportedRole };
  onLoginChange: (field: 'email' | 'password', value: string) => void;
  onQuickFill: (email: string, password: string) => void;
  onRegisterChange: (field: 'fullName' | 'email' | 'password' | 'role', value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  onRegisterSubmit: (event: FormEvent<HTMLFormElement>) => Promise<void>;
}) {
  return (
    <main className="login-shell">
      <div className="login-grid">
        <section className="login-card login-hero">
          <div className="login-badge mt-6">HEXAMIND SECURITY</div>
          <a href="http://localhost:8010/" className="login-hero-logo" aria-label="IoTinel Security Platform" style={{ textDecoration: 'none' }}>
            <div className="login-hero-logo-icon">
              <svg width="58" height="58" viewBox="0 0 42 42" xmlns="http://www.w3.org/2000/svg">
                <rect width="42" height="42" rx="12" fill="#0a2a1e" />
                <rect width="42" height="42" rx="12" fill="url(#analystHeroLogoGrad)" opacity="0.7" />
                <defs>
                  <linearGradient id="analystHeroLogoGrad" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="#1ec98f" />
                    <stop offset="100%" stopColor="#0fa872" />
                  </linearGradient>
                </defs>
                <path d="M21 8 L32 13 L32 21 Q32 29 21 34 Q10 29 10 21 L10 13 Z" fill="none" stroke="#1ec98f" strokeWidth="1.5" strokeLinejoin="round" />
                <circle cx="21" cy="21" r="4" fill="#1ec98f" opacity="0.9" />
                <circle cx="21" cy="21" r="2" fill="#030d0a" />
                <circle cx="22.2" cy="19.8" r="0.8" fill="#1ec98f" />
              </svg>
            </div>
            <div className="login-hero-logo-text">
              <div className="login-hero-logo-name">
                <span className="iot">IoT</span>
                <span className="inel">inel</span>
              </div>
              <div className="login-hero-logo-sub">Security Platform</div>
            </div>
          </a>
          <p className="login-hero-copy">
            One mission-critical platform for live intrusion detection, model operations, drift monitoring, and secure administration.
          </p>

          <div className="login-role-grid">
            {QUICK_ACCOUNTS.map((account) => (
              <article key={account.role} className="login-role-card">
                <div className={`login-role-title ${account.tone}`}>{account.label}</div>
                <p className="login-role-copy">{account.copy}</p>
              </article>
            ))}
          </div>

          <div className="login-shortcuts">
            {QUICK_ACCOUNTS.map((account) => (
              <button
                key={account.role}
                type="button"
                className={`role-chip ${account.tone} ${email === account.email ? 'active' : ''}`}
                onClick={() => onQuickFill(account.email, account.password)}
              >
                {account.label}
              </button>
            ))}
          </div>
        </section>

        <div className="login-side-stack">
          <section className="login-card login-panel">
            <h2 className="login-panel-title">Login</h2>
            <form className="mt-8 space-y-4" onSubmit={onSubmit}>
              <input
                className="field field-strong"
                name="email"
                value={email}
                onChange={(event) => onLoginChange('email', event.target.value)}
                autoComplete="username"
                placeholder="analyst@hexamind.local"
              />
              <input
                className="field field-strong"
                name="password"
                type="password"
                value={password}
                onChange={(event) => onLoginChange('password', event.target.value)}
                autoComplete="current-password"
                placeholder="Password"
              />
              {error ? <div className="inline-alert">{error}</div> : null}
              <button className="button-primary button-emerald flex w-full items-center justify-center gap-3" disabled={loading} type="submit">
                {loading ? <span className="loading-dot" /> : null}
                {loading ? 'Signing in...' : 'Enter Security Analyst Workspace'}
              </button>
            </form>
          </section>

          <section className="login-card login-panel">
            <h2 className="login-panel-title">Register</h2>
            <form className="mt-8 space-y-4" onSubmit={onRegisterSubmit}>
              <input
                className="field"
                name="full_name"
                value={registerForm.fullName}
                onChange={(event) => onRegisterChange('fullName', event.target.value)}
                placeholder="full name"
              />
              <input
                className="field field-strong"
                name="register_email"
                value={registerForm.email}
                onChange={(event) => onRegisterChange('email', event.target.value)}
                placeholder="scientist@hexamind.local"
              />
              <input
                className="field field-strong"
                name="register_password"
                type="password"
                value={registerForm.password}
                onChange={(event) => onRegisterChange('password', event.target.value)}
                placeholder="password (min 6 characters)"
              />
              <select
                className="field"
                name="role"
                value={registerForm.role}
                onChange={(event) => onRegisterChange('role', event.target.value)}
              >
                <option value="security_analyst">Security Analyst</option>
                <option value="data_scientist">Data Scientist</option>
                <option value="administrator">Administrator</option>
              </select>
              {registerError ? <div className="inline-alert">{registerError}</div> : null}
              {registerSuccess ? <div className="inline-success">{registerSuccess}</div> : null}
              <button className="button-secondary w-full" disabled={registerLoading} type="submit">
                {registerLoading ? 'Submitting...' : 'Request Access'}
              </button>
            </form>
          </section>
        </div>
      </div>
    </main>
  );
}

export default function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const [authReady, setAuthReady] = useState(false);
  const [user, setUser] = useState<AuthUser | null>(() => readStoredUser());
  const [loginError, setLoginError] = useState('');
  const [loginLoading, setLoginLoading] = useState(false);
  const [loginForm, setLoginForm] = useState({ email: 'analyst@hexamind.local', password: 'analyst123' });
  const [registerForm, setRegisterForm] = useState({
    fullName: '',
    email: 'scientist@hexamind.local',
    password: 'scientist123',
    role: 'security_analyst' as SupportedRole,
  });
  const [registerError, setRegisterError] = useState('');
  const [registerSuccess, setRegisterSuccess] = useState('');
  const [registerLoading, setRegisterLoading] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [datasets, setDatasets] = useState<DatasetOption[]>(ANALYST_FALLBACK_DATASETS);
  const [dataset, setDataset] = useState('eMBB');
  const [features, setFeatures] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [aiExplanation, setAiExplanation] = useState('');
  const [aiExplanationLoading, setAiExplanationLoading] = useState(false);
  const [batchResults, setBatchResults] = useState<Array<Record<string, unknown>>>([]);
  const [summaryRows, setSummaryRows] = useState<SummaryRow[]>([]);
  const [timeline, setTimeline] = useState<TimelineRow[]>([]);
  const [attacks, setAttacks] = useState<Record<string, unknown>>({});
  const [history, setHistory] = useState<HistoryRow[]>([]);

  const currentPath = useMemo(() => normalizePath(location.pathname), [location.pathname]);
  const isLoginPage = currentPath === '/' || currentPath === LOGIN_PATH;
  const activePage = isLoginPage ? ROLE_CONFIG.homePath : currentPath;
  const selected = datasets.find((item) => item.dataset === dataset) ?? ANALYST_FALLBACK_DATASETS.find((item) => item.dataset === dataset) ?? ANALYST_FALLBACK_DATASETS[0];
  const attackTypes = (attacks.attack_types || {}) as Record<string, number>;
  const topAttack = Object.entries(attackTypes).sort((left, right) => right[1] - left[1])[0];
  const totalPredictions = Number(summaryRows[0]?.total_predictions || 0);
  const attackRate = totalPredictions ? (Number(summaryRows[0]?.malicious_count || 0) / totalPredictions) * 100 : 0;
  const maliciousCount = Number(summaryRows[0]?.malicious_count || 0);
  const benignCount = Math.max(totalPredictions - maliciousCount, 0);
  const timelineChart = timeline
    .slice(-8)
    .map((point) => Number(point.total || point.count || 0));
  const attackBreakdown = Object.entries(attackTypes)
    .sort((left, right) => right[1] - left[1])
    .slice(0, 5)
    .map(([label, value]) => ({ label, value: Number(value) }));
  const historySummary = history.slice(0, 8);
  const avgConfidence = historySummary.length
    ? historySummary.reduce((sum, row) => sum + Number(row.confidence || 0), 0) / historySummary.length
    : Number(prediction?.confidence || 0);
  const historyTrend = historySummary
    .map((row) => Number(row.confidence || 0) * 100)
    .reverse();
  const predictionView = derivePredictionViewModel(dataset, prediction, features);

  const handleUnauthorized = (message: string) => {
    writeStoredUser(null);
    setUser(null);
    setError('');
    setLoginError(message);
    window.location.href = GATEWAY_LOGIN_URL;
  };

  const refreshAnalystData = async () => {
    setLoading(true);
    // Fetch datasets separately so a failure doesn't prevent the page from rendering inputs
    try {
      const datasetPayload = await api<DatasetOption[]>('/detect/datasets');
      const resolvedDatasets = ANALYST_FALLBACK_DATASETS.map((fallback) => {
        const live = datasetPayload.find((item) => item.dataset === fallback.dataset);
        return (live && live.features?.length) ? live : fallback;
      });
      setDatasets(resolvedDatasets);
      if (resolvedDatasets[0] && !resolvedDatasets.some((item) => item.dataset === dataset)) {
        setDataset(resolvedDatasets[0].dataset);
      }
    } catch {
      // Keep the fallback datasets already set in initial state
    }
    const [summaryPayload, attacksPayload, timelinePayload, historyPayload] = await Promise.all([
      api<{ summary: SummaryRow[] }>('/dashboard/summary'),
      api<Record<string, unknown>>('/dashboard/attacks'),
      api<{ timeline: TimelineRow[] }>('/dashboard/timeline'),
      api<HistoryRow[]>('/detect/history'),
    ]);
    setSummaryRows(Array.isArray(summaryPayload.summary) ? summaryPayload.summary : []);
    setAttacks(attacksPayload);
    setTimeline(Array.isArray(timelinePayload.timeline) ? timelinePayload.timeline : []);
    setHistory(Array.isArray(historyPayload) ? historyPayload : []);
    setLoading(false);
  };

  useEffect(() => {
    if (isLoginPage) {
      setAuthReady(true);
      setLoading(false);
      return;
    }
    const storedUser = readStoredUser();
    if (storedUser && storedUser.role !== ROLE_CONFIG.role) {
      window.location.href = redirectForRole(storedUser.role);
      return;
    }
    refreshAnalystData()
      .then(() => {
        setError('');
        setAuthReady(true);
      })
      .catch((err: Error) => {
        if (isAuthError(err.message)) {
          handleUnauthorized(err.message);
          return;
        }
        setError(err.message);
        setLoading(false);
        setAuthReady(true);
      });
  }, [isLoginPage]);

  useEffect(() => {
    if (!selected) return;
    const next: Record<string, string> = {};
    selected.features.forEach((feature) => {
      next[feature] = features[feature] || '';
    });
    setFeatures(next);
  }, [dataset, selected?.dataset]);

  const submitLogin = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const email = loginForm.email.trim();
    const password = loginForm.password;
    setLoginError('');
    setLoginLoading(true);
    try {
      const payload = await api<AuthResponse>('/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      writeStoredUser(payload.user);
      setUser(payload.user);
      if (payload.user.role !== ROLE_CONFIG.role) {
        window.location.href = redirectForRole(payload.user.role);
        return;
      }
      navigate(HOME_PATH, { replace: true });
    } catch (err) {
      setLoginError((err as Error).message);
    } finally {
      setLoginLoading(false);
    }
  };

  const submitRegister = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setRegisterError('');
    setRegisterSuccess('');
    setRegisterLoading(true);
    try {
      const payload = await api<RegistrationResponse>('/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          full_name: registerForm.fullName.trim(),
          email: registerForm.email.trim(),
          password: registerForm.password,
          role: registerForm.role,
        }),
      });
      setRegisterSuccess(payload.message);
      setRegisterForm((current) => ({ ...current, fullName: '', password: '' }));
    } catch (err) {
      setRegisterError((err as Error).message);
    } finally {
      setRegisterLoading(false);
    }
  };

  const logout = () => {
    writeStoredUser(null);
    setUser(null);
    window.location.href = GATEWAY_LOGIN_URL;
  };

  const submitPrediction = async () => {
    if (!selected) return;
    setError('');
    setAiExplanation('');
    setAiExplanationLoading(false);
    try {
      const payload = Object.fromEntries(
        Object.entries(features).map(([key, value]) => [key, value === '' ? null : Number.isNaN(Number(value)) ? value : Number(value)]),
      );
      const predictionPayload = await api<PredictionResponse>('/detect/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset, features: payload }),
      });
      setPrediction(predictionPayload);
      const viewModel = derivePredictionViewModel(dataset, predictionPayload, payload);
      if (viewModel.label === 'Benign') {
        setAiExplanation('Traffic classified as benign — no threat explanation needed.');
      } else {
        setAiExplanationLoading(true);
        try {
          const explanationText = await getCohereExplanation(
            dataset,
            viewModel.attackType,
            viewModel.label,
            viewModel.confidence,
            payload,
          );
          setAiExplanation(explanationText);
        } finally {
          setAiExplanationLoading(false);
        }
      }
      await refreshAnalystData();
      navigate(routeFor('/live-detection'));
    } catch (err) {
      const message = (err as Error).message;
      if (isAuthError(message)) {
        handleUnauthorized(message);
        return;
      }
      setError(message);
    }
  };

  const uploadBatch = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setError('');
    try {
      const form = new FormData();
      form.append('file', file);
      const response = await fetch(`${API_BASE}/detect/batch?dataset=${encodeURIComponent(dataset)}`, {
        method: 'POST',
        body: form,
        credentials: 'include',
      });
      const raw = await response.text();
      let payload: Record<string, unknown> = {};
      if (raw) {
        try {
          payload = JSON.parse(raw) as Record<string, unknown>;
        } catch {
          payload = { detail: raw };
        }
      }
      if (!response.ok) {
        throw new Error(String(payload.detail || 'Batch prediction failed'));
      }
      setBatchResults(Array.isArray(payload.results) ? payload.results : []);
      navigate(routeFor('/batch-analysis'));
    } catch (err) {
      const message = (err as Error).message;
      if (isAuthError(message)) {
        handleUnauthorized(message);
        return;
      }
      setError(message);
    }
  };

  const exportBatch = () => {
    const content = ['index,prediction,attack_type,confidence'];
    batchResults.forEach((row, index) => {
      content.push(`${index},${row.prediction || row.verdict || ''},${row.attack_type || ''},${row.confidence || ''}`);
    });
    const blob = new Blob([content.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${dataset}-batch-results.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  if (!authReady && !isLoginPage) {
    return (
      <main className="login-shell">
        <div className="login-card flex items-center gap-4">
          <span className="loading-dot" />
          <div>
            <div className="text-lg font-semibold">Loading analyst workspace</div>
            <div className="mt-1 text-sm text-slate-400">Restoring session and fetching telemetry.</div>
          </div>
        </div>
      </main>
    );
  }

  if (isLoginPage) {
    return (
      <LoginView
        email={loginForm.email}
        password={loginForm.password}
        error={loginError}
        loading={loginLoading}
        registerError={registerError}
        registerLoading={registerLoading}
        registerSuccess={registerSuccess}
        registerForm={registerForm}
        onLoginChange={(field, value) => {
          setLoginForm((current) => ({ ...current, [field]: value }));
          if (loginError) setLoginError('');
        }}
        onQuickFill={(email, password) => {
          setLoginForm({ email, password });
          setLoginError('');
        }}
        onRegisterChange={(field, value) => {
          setRegisterForm((current) => ({ ...current, [field]: value }));
          if (registerError) setRegisterError('');
          if (registerSuccess) setRegisterSuccess('');
        }}
        onSubmit={submitLogin}
        onRegisterSubmit={submitRegister}
      />
    );
  }

  const pageTitleMap: Record<string, { title: string; subtitle: string }> = {
    '/dashboard': {
      title: 'Operational Dashboard',
      subtitle: 'Review current prediction volume, attack mix, and the latest telemetry timeline.',
    },
    '/live-detection': {
      title: 'Live Detection',
      subtitle: 'Submit single traffic slices for prediction and inspect explanation details immediately.',
    },
    '/batch-analysis': {
      title: 'Batch Analysis',
      subtitle: 'Upload CSV samples and export the scored batch results through the detection service.',
    },
    '/model-comparison': {
      title: 'Prediction History',
      subtitle: 'Browse recently scored requests and compare recent outcomes over time.',
    },
    '/swagger': {
      title: 'API References',
      subtitle: 'Quick access to the main backend documentation routes used by the analyst tools.',
    },
  };
  const pageCopy = pageTitleMap[activePage] || pageTitleMap['/dashboard'];

  return (
    <Shell activePath={activePage} onLogout={logout} subtitle={pageCopy.subtitle} title={pageCopy.title} user={user}>
      {error ? <div className="inline-alert">{error}</div> : null}

      {activePage === '/dashboard' ? (
        <>
          <section className="grid gap-4 md:grid-cols-3">
            <div className="metric-card">
              <div className="text-sm text-slate-400">Total Predictions</div>
              <div className="mt-3 text-3xl font-semibold">{totalPredictions}</div>
            </div>
            <div className="metric-card">
              <div className="text-sm text-slate-400">Attack Rate</div>
              <div className="mt-3 text-3xl font-semibold">{attackRate.toFixed(1)}%</div>
            </div>
            <div className="metric-card">
              <div className="text-sm text-slate-400">Top Attack Type</div>
              <div className="mt-3 text-3xl font-semibold">{topAttack?.[0] || 'n/a'}</div>
            </div>
          </section>

          <section className="grid gap-6 lg:grid-cols-2">
            <SectionCard title="Timeline">
              {loading ? (
                <div className="h-48 rounded-2xl skeleton" />
              ) : timeline.length ? (
                <div className="space-y-4">
                  <MiniAreaChart 
                    points={timelineChart} 
                    title="Prediction Volume Over Time"
                    xAxisLabel="Time Period"
                    yAxisLabel="Predictions Count"
                  />
                  <SparkBars
                    items={timeline.slice(-6).map((point, index) => ({
                      label: String(point.time_bucket || point.bucket || point.timestamp || `t-${index}`),
                      value: Number(point.total || point.count || 0),
                    }))}
                    tone="cyan"
                  />
                </div>
              ) : (
                <div className="empty-state">No timeline data yet. Run a prediction to populate telemetry.</div>
              )}
            </SectionCard>

            <SectionCard title="Attack Distribution">
              {loading ? (
                <div className="h-48 rounded-2xl skeleton" />
              ) : Object.entries(attackTypes).length ? (
                <div className="space-y-4">
                  <div className="insight-grid">
                    <StatTile label="Malicious" value={String(maliciousCount)} hint={`${attackRate.toFixed(1)}% of traffic`} />
                    <StatTile label="Benign" value={String(benignCount)} hint="Estimated from dashboard totals" />
                  </div>
                  <SparkBars items={attackBreakdown} tone="amber" />
                </div>
              ) : (
                <div className="empty-state">No attack categories have been recorded yet.</div>
              )}
            </SectionCard>
          </section>
        </>
      ) : null}

      {activePage === '/live-detection' ? (
        <section className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <SectionCard title="Single Prediction" action={<div className="status-pill success">{dataset}</div>}>
            <div>
              <label className="text-sm font-medium text-slate-300">Dataset</label>
              <select value={dataset} onChange={(event) => setDataset(event.target.value)} className="field mt-2">
                {datasets.map((item) => (
                  <option key={item.dataset}>{item.dataset}</option>
                ))}
              </select>
            </div>
            <div className="mt-5 grid gap-3 md:grid-cols-2">
              {(selected?.features || ['Dur', 'TotPkts', 'TotBytes', 'Rate', 'Load', 'Loss', 'pLoss', 'TcpRtt']).map((feature) => (
                <label key={feature} className="block">
                  <span className="mb-2 block text-sm text-slate-400">{feature}</span>
                  <input
                    className="field"
                    value={features[feature] || ''}
                    onChange={(event) => setFeatures((current) => ({ ...current, [feature]: event.target.value }))}
                    placeholder={`Enter ${feature}`}
                  />
                </label>
              ))}
            </div>
            <div className="mt-5 flex flex-col gap-3 md:flex-row">
              <button
                className="button-secondary"
                onClick={() => {
                  if (!selected) return;
                  setFeatures((current) => ({
                    ...current,
                    ...buildRandomFeatureValues(dataset, selected.features),
                  }));
                }}
              >
                Generate Values
              </button>
              <button className="button-primary" onClick={submitPrediction}>
                Predict and explain
              </button>
            </div>
          </SectionCard>

          <SectionCard title="Prediction Result">
            {!prediction ? (
              <div className="empty-state">Submit a sample to see the label, confidence, and AI explanation.</div>
            ) : (
              <div className="space-y-4 text-sm">
                <div className="insight-grid">
                  <StatTile label="Confidence" value={`${(Number(prediction.confidence || 0) * 100).toFixed(2)}%`} hint="Current model certainty" />
                  <StatTile label="Recent Avg" value={`${(avgConfidence * 100).toFixed(2)}%`} hint="Latest prediction history average" />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Label</span>
                  <strong>{predictionView.label}</strong>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Attack Type</span>
                  <span className="status-pill danger">{predictionView.attackType || 'n/a'}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Severity</span>
                  <span className="status-pill warning">{String(prediction.severity || 'n/a')}</span>
                </div>
                <div>
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-slate-400">Confidence</span>
                    <strong>{(Number(prediction.confidence || 0) * 100).toFixed(2)}%</strong>
                  </div>
                  <div className="h-3 overflow-hidden rounded-full bg-slate-800">
                    <div className="h-full bg-cyan-400 transition-all duration-300" style={{ width: `${Number(prediction.confidence || 0) * 100}%` }} />
                  </div>
                </div>
                <div>
                  <div className="mb-3 text-sm font-semibold">AI Explanation</div>
                  <div className="surface-soft px-4 py-3 leading-7 text-slate-200">
                    {aiExplanationLoading ? 'Generating AI explanation...' : aiExplanation || 'AI explanation unavailable'}
                  </div>
                </div>
                <div>
                  <MiniAreaChart 
                    points={historyTrend.length ? historyTrend : [Number(prediction.confidence || 0) * 100]} 
                    title="Recent Confidence Trend"
                    xAxisLabel="Prediction Number"
                    yAxisLabel="Confidence (%)"
                  />
                </div>
              </div>
            )}
          </SectionCard>
        </section>
      ) : null}

      {activePage === '/batch-analysis' ? (
        <SectionCard
          action={
            <button className="button-secondary" onClick={exportBatch}>
              Export CSV
            </button>
          }
          title="Batch Prediction"
        >
          {batchResults.length ? (
            <div className="insight-grid mb-5">
              <StatTile label="Rows Scored" value={String(batchResults.length)} />
              <StatTile
                label="Avg Confidence"
                value={`${(
                  (batchResults.reduce((sum, row) => sum + Number(row.confidence || 0), 0) / Math.max(batchResults.length, 1)) * 100
                ).toFixed(2)}%`}
              />
            </div>
          ) : null}
          <div className="rounded-2xl border border-dashed border-white/10 bg-slate-950/50 p-5">
            <div className="text-sm text-slate-400">Upload a CSV for the selected dataset and score it through the same detection gateway.</div>
            <input accept=".csv" className="mt-4 block text-sm" onChange={uploadBatch} type="file" />
          </div>
          <div className="mt-6 overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Prediction</th>
                  <th>Attack Type</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {batchResults.length ? (
                  batchResults.map((row, index) => (
                    <tr key={index}>
                      <td>{String(row.prediction || row.verdict || 'n/a')}</td>
                      <td>{String(row.attack_type || 'n/a')}</td>
                      <td>{Number(row.confidence || 0).toFixed(4)}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={3}>
                      <div className="empty-state">No batch results yet. Upload a CSV file to populate this table.</div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </SectionCard>
      ) : null}

      {activePage === '/model-comparison' ? (
        <SectionCard title="Prediction History">
          {history.length ? (
            <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr] mb-6">
              <div className="surface-soft p-4">
                <MiniAreaChart 
                  points={historyTrend} 
                  title="Recent Confidence Curve"
                  xAxisLabel="Prediction Number"
                  yAxisLabel="Confidence (%)"
                />
              </div>
              <div className="surface-soft p-4">
                <div className="mb-3 text-sm font-semibold text-slate-200">Latest Attack Types</div>
                <SparkBars
                  items={historySummary.map((row, index) => ({
                    label: String(row.attack_type || row.prediction || `row-${index}`),
                    value: Math.round(Number(row.confidence || 0) * 100),
                  }))}
                  tone="emerald"
                />
              </div>
            </div>
          ) : null}
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Prediction</th>
                  <th>Confidence</th>
                  <th>Attack Type</th>
                  <th>Requested By</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {history.length ? (
                  history.map((row, index) => (
                    <tr key={index}>
                      <td>{String(row.dataset || 'n/a')}</td>
                      <td>{String(row.prediction || 'n/a')}</td>
                      <td>{Number(row.confidence || 0).toFixed(4)}</td>
                      <td>{String(row.attack_type || 'n/a')}</td>
                      <td>{String(row.requested_by || 'n/a')}</td>
                      <td>{String(row.created_at || 'n/a')}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={6}>
                      <div className="empty-state">No prediction history yet. Run a live prediction first.</div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </SectionCard>
      ) : null}

      {activePage === '/swagger' ? (
        <section className="grid gap-4 md:grid-cols-3">
          <a className="surface-card p-6 transition-transform duration-150 hover:-translate-y-1" href="http://localhost:8002/docs" rel="noreferrer" target="_blank">
            <div className="text-sm uppercase tracking-[0.24em] text-cyan-300">API</div>
            <div className="mt-2 text-xl font-semibold">Detection Swagger</div>
          </a>
          <a className="surface-card p-6 transition-transform duration-150 hover:-translate-y-1" href="http://localhost:8005/docs" rel="noreferrer" target="_blank">
            <div className="text-sm uppercase tracking-[0.24em] text-cyan-300">API</div>
            <div className="mt-2 text-xl font-semibold">Dashboard Swagger</div>
          </a>
          <a className="surface-card p-6 transition-transform duration-150 hover:-translate-y-1" href="http://localhost:8088/docs" rel="noreferrer" target="_blank">
            <div className="text-sm uppercase tracking-[0.24em] text-cyan-300">API</div>
            <div className="mt-2 text-xl font-semibold">MLOPS Swagger</div>
          </a>
        </section>
      ) : null}
    </Shell>
  );
}
