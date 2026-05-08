import type { HeatmapDatum } from '@/types/dashboard';

export function HeatmapGrid({ data }: { data: HeatmapDatum[] }) {
  return (
    <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-6">
      <div className="mb-6">
        <p className="text-sm uppercase tracking-[0.2em] text-fuchsia-300">Heatmap</p>
        <h3 className="text-xl font-semibold text-white">Resistance hotspot matrix</h3>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {data.map((entry) => {
          const intensity = Math.min(1, Math.max(0.1, entry.resistanceRate));
          return (
            <div
              key={`${entry.microbe}-${entry.antibiotic}`}
              className="rounded-2xl border border-white/10 p-4"
              style={{ backgroundColor: `rgba(34,211,238,${intensity * 0.45})` }}
            >
              <p className="text-sm font-medium text-white">{entry.microbe}</p>
              <p className="text-xs text-slate-200">{entry.antibiotic}</p>
              <p className="mt-3 text-lg font-semibold text-white">{Math.round(entry.resistanceRate * 100)}%</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
