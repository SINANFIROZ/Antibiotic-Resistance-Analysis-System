import { HeatmapGrid } from '@/components/heatmap-grid';
import { StatCard } from '@/components/stat-card';
import { TrendChart } from '@/components/trend-chart';

const metrics = [
  { label: 'Longitudinal cohorts', value: '12', delta: '+2 curated this month' },
  { label: 'Dataset comparisons', value: '37', delta: '5 active analyses' },
  { label: 'Export-ready insights', value: '94', delta: '+13 new artifacts' },
  { label: 'Forecast confidence', value: '76%', delta: '+4.8% calibration gain' },
];

const trend = [
  { name: 'Week 1', resistant: 31, susceptible: 58 },
  { name: 'Week 2', resistant: 33, susceptible: 55 },
  { name: 'Week 3', resistant: 37, susceptible: 51 },
  { name: 'Week 4', resistant: 40, susceptible: 48 },
  { name: 'Week 5', resistant: 44, susceptible: 46 },
];

const heatmap = [
  { microbe: 'E. coli', antibiotic: 'Fosfomycin', resistanceRate: 0.18 },
  { microbe: 'K. pneumoniae', antibiotic: 'Meropenem', resistanceRate: 0.49 },
  { microbe: 'P. aeruginosa', antibiotic: 'Ceftolozane-Tazobactam', resistanceRate: 0.42 },
  { microbe: 'S. aureus', antibiotic: 'Clindamycin', resistanceRate: 0.51 },
  { microbe: 'A. baumannii', antibiotic: 'Tigecycline', resistanceRate: 0.38 },
  { microbe: 'E. faecalis', antibiotic: 'Linezolid', resistanceRate: 0.14 },
];

export default function ResearchDashboardPage() {
  return (
    <div className="space-y-6">
      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {metrics.map((metric) => (
          <StatCard key={metric.label} {...metric} />
        ))}
      </section>
      <section className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <TrendChart data={trend} />
        <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-6">
          <p className="text-sm uppercase tracking-[0.2em] text-fuchsia-300">Research tools</p>
          <h3 className="mt-2 text-xl font-semibold text-white">Comparative surveillance analysis</h3>
          <ul className="mt-5 space-y-3 text-sm text-slate-300">
            <li className="rounded-2xl border border-white/10 bg-white/5 p-4">Cohort comparison with geographic and temporal context</li>
            <li className="rounded-2xl border border-white/10 bg-white/5 p-4">Feature importance and explainability-ready model comparisons</li>
            <li className="rounded-2xl border border-white/10 bg-white/5 p-4">Export pipeline for CSV/PDF outputs and publication support</li>
          </ul>
        </div>
      </section>
      <HeatmapGrid data={heatmap} />
    </div>
  );
}
