'use client';

import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import type { TrendDatum } from '@/types/dashboard';

export function TrendChart({ data }: { data: TrendDatum[] }) {
  return (
    <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-6">
      <div className="mb-6">
        <p className="text-sm uppercase tracking-[0.2em] text-cyan-300">Trend forecast</p>
        <h3 className="text-xl font-semibold text-white">Resistance trajectory</h3>
      </div>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="resistantFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.45} />
                <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.02} />
              </linearGradient>
              <linearGradient id="susceptibleFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#34d399" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#34d399" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(148,163,184,0.15)" vertical={false} />
            <XAxis dataKey="name" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip />
            <Area type="monotone" dataKey="resistant" stroke="#22d3ee" fill="url(#resistantFill)" strokeWidth={2} />
            <Area type="monotone" dataKey="susceptible" stroke="#34d399" fill="url(#susceptibleFill)" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
