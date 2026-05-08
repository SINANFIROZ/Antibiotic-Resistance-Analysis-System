import { StatCard } from '@/components/stat-card';
import { PredictionWorkspace } from '@/components/prediction-workspace';

const metrics = [
  { label: 'Today\'s pending isolates', value: '18', delta: '3 urgent stewardship reviews' },
  { label: 'Average AI confidence', value: '82%', delta: '+5.4% from prior week' },
  { label: 'Recommended therapy shifts', value: '11', delta: '7 de-escalation opportunities' },
];

export default function DoctorDashboardPage() {
  return (
    <div className="space-y-6">
      <section className="grid gap-4 md:grid-cols-3">
        {metrics.map((metric) => (
          <StatCard key={metric.label} {...metric} />
        ))}
      </section>
      <PredictionWorkspace />
    </div>
  );
}
