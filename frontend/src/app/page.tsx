import Link from 'next/link';
import { ArrowRight, BrainCircuit, ChartNoAxesCombined, ShieldCheck } from 'lucide-react';

import { StatCard } from '@/components/stat-card';

const metrics = [
  { label: 'Clinical predictions / day', value: '1,248', delta: '+18.2% vs last month' },
  { label: 'High-risk resistance alerts', value: '73', delta: '+6.1% surveillance delta' },
  { label: 'Active facilities onboarded', value: '26', delta: '+4 new networks' },
];

export default function Home() {
  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(34,211,238,0.15),_transparent_35%),radial-gradient(circle_at_right,_rgba(168,85,247,0.15),_transparent_30%),linear-gradient(180deg,_#020617,_#0f172a)] text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4 py-6 lg:px-6">
        <header className="flex items-center justify-between rounded-full border border-white/10 bg-white/5 px-5 py-3 backdrop-blur">
          <div>
            <p className="text-xs uppercase tracking-[0.35em] text-cyan-300">AMR Intelligence Platform</p>
          </div>
          <div className="flex items-center gap-3 text-sm">
            <Link href="/login" className="rounded-full px-4 py-2 text-slate-300 transition hover:bg-white/5 hover:text-white">Login</Link>
            <Link href="/signup" className="rounded-full bg-cyan-400 px-4 py-2 font-medium text-slate-950 transition hover:bg-cyan-300">Get started</Link>
          </div>
        </header>

        <section className="grid flex-1 items-center gap-12 py-16 lg:grid-cols-[1.15fr_0.85fr]">
          <div>
            <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">AI-powered antimicrobial stewardship</p>
            <h1 className="mt-6 max-w-4xl text-5xl font-semibold leading-tight text-white sm:text-6xl">
              Production-grade AMR intelligence for clinicians, labs, and surveillance teams.
            </h1>
            <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-300">
              Transform isolate data into explainable treatment insights, resistance forecasts, stewardship alerts, and research-ready analytics with a modern enterprise healthcare interface.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link href="/dashboard/doctor" className="inline-flex items-center gap-2 rounded-full bg-cyan-400 px-5 py-3 font-semibold text-slate-950 transition hover:bg-cyan-300">
                Launch platform
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link href="/dashboard/research" className="rounded-full border border-white/10 px-5 py-3 font-semibold text-white transition hover:bg-white/5">
                Explore analytics
              </Link>
            </div>
            <div className="mt-10 grid gap-4 md:grid-cols-3">
              {metrics.map((metric) => (
                <StatCard key={metric.label} {...metric} />
              ))}
            </div>
          </div>
          <div className="rounded-[2rem] border border-white/10 bg-slate-950/70 p-6 shadow-2xl shadow-cyan-950/20 backdrop-blur">
            <div className="grid gap-4">
              <div className="rounded-3xl border border-cyan-400/20 bg-cyan-400/10 p-5">
                <div className="flex items-center gap-3">
                  <BrainCircuit className="h-6 w-6 text-cyan-300" />
                  <div>
                    <p className="text-lg font-semibold text-white">Explainable decision support</p>
                    <p className="text-sm text-slate-200">Confidence scoring, feature reasoning, and ranked antibiotic alternatives.</p>
                  </div>
                </div>
              </div>
              <div className="rounded-3xl border border-fuchsia-400/20 bg-fuchsia-400/10 p-5">
                <div className="flex items-center gap-3">
                  <ChartNoAxesCombined className="h-6 w-6 text-fuchsia-300" />
                  <div>
                    <p className="text-lg font-semibold text-white">Research-grade surveillance</p>
                    <p className="text-sm text-slate-200">Heatmaps, temporal resistance signals, cohort trends, and export-ready analytics.</p>
                  </div>
                </div>
              </div>
              <div className="rounded-3xl border border-emerald-400/20 bg-emerald-400/10 p-5">
                <div className="flex items-center gap-3">
                  <ShieldCheck className="h-6 w-6 text-emerald-300" />
                  <div>
                    <p className="text-lg font-semibold text-white">Enterprise controls</p>
                    <p className="text-sm text-slate-200">JWT auth, RBAC, audit logging, PDF reporting, and cloud-native deployment.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
