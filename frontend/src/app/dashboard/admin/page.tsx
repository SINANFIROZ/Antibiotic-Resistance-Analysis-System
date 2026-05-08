import { StatCard } from '@/components/stat-card';
import { HeatmapGrid } from '@/components/heatmap-grid';
import { TrendChart } from '@/components/trend-chart';

const metrics = [
  { label: 'Tenants monitored', value: '26', delta: '+4 this quarter' },
  { label: 'Model versions governed', value: '14', delta: '2 pending approval' },
  { label: 'Audit events today', value: '3,842', delta: '+11.6% operational load' },
  { label: 'Report exports', value: '418', delta: '+21.4% clinician adoption' },
];

const trend = [
  { name: 'Jan', resistant: 18, susceptible: 41 },
  { name: 'Feb', resistant: 22, susceptible: 39 },
  { name: 'Mar', resistant: 28, susceptible: 43 },
  { name: 'Apr', resistant: 33, susceptible: 40 },
  { name: 'May', resistant: 37, susceptible: 38 },
];

const heatmap = [
  { microbe: 'E. coli', antibiotic: 'Ciprofloxacin', resistanceRate: 0.71 },
  { microbe: 'K. pneumoniae', antibiotic: 'Ceftriaxone', resistanceRate: 0.66 },
  { microbe: 'P. aeruginosa', antibiotic: 'Meropenem', resistanceRate: 0.58 },
  { microbe: 'A. baumannii', antibiotic: 'Colistin', resistanceRate: 0.27 },
  { microbe: 'S. aureus', antibiotic: 'Oxacillin', resistanceRate: 0.61 },
  { microbe: 'E. faecium', antibiotic: 'Vancomycin', resistanceRate: 0.35 },
];

export default function AdminDashboardPage() {
  return (
    <div className="space-y-6">
      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {metrics.map((metric) => (
          <StatCard key={metric.label} {...metric} />
        ))}
      </section>
      <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <TrendChart data={trend} />
        <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-6">
          <p className="text-sm uppercase tracking-[0.2em] text-emerald-300">Governance</p>
          <h3 className="mt-2 text-xl font-semibold text-white">Operational control tower</h3>
          <div className="mt-6 space-y-4 text-sm text-slate-300">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="font-medium text-white">Security posture</p>
              <p className="mt-2">JWT authentication, role segmentation, audit trails, and rate limiting are active across the API perimeter.</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="font-medium text-white">Deployment readiness</p>
              <p className="mt-2">Dockerized frontend/backend/postgres topology is ready for cloud lift-and-shift deployment.</p>
            </div>
          </div>
        </div>
      </section>
      <HeatmapGrid data={heatmap} />
    </div>
  );
}
