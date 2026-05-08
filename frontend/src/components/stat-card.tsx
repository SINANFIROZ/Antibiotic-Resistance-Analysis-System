'use client';

import { motion } from 'framer-motion';
import { Activity } from 'lucide-react';

export type StatCardProps = {
  label: string;
  value: string;
  delta: string;
};

export function StatCard({ label, value, delta }: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className="rounded-3xl border border-white/10 bg-slate-950/70 p-5 shadow-2xl shadow-cyan-950/10 backdrop-blur"
    >
      <div className="mb-4 flex items-center justify-between">
        <span className="text-sm text-slate-400">{label}</span>
        <Activity className="h-4 w-4 text-cyan-300" />
      </div>
      <div className="space-y-1">
        <p className="text-3xl font-semibold text-white">{value}</p>
        <p className="text-sm text-emerald-300">{delta}</p>
      </div>
    </motion.div>
  );
}
