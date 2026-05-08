import { PredictionWorkspace } from '@/components/prediction-workspace';

export default function PredictionsPage() {
  return (
    <div className="space-y-6">
      <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-6 text-slate-300">
        <p className="text-sm uppercase tracking-[0.2em] text-cyan-300">Prediction operations</p>
        <h3 className="mt-2 text-2xl font-semibold text-white">Explainable inference workspace</h3>
        <p className="mt-3 max-w-3xl leading-7">
          Run AMR predictions, review ranked alternatives, and export patient-facing or stewardship-facing reports with auditability.
        </p>
      </div>
      <PredictionWorkspace />
    </div>
  );
}
