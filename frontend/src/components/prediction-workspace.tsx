'use client';

import { motion } from 'framer-motion';
import { Beaker, ShieldCheck } from 'lucide-react';
import { useState } from 'react';

const microbes = ['Escherichia coli', 'Klebsiella pneumoniae', 'Pseudomonas aeruginosa'];
const antibiotics = ['Meropenem', 'Ceftriaxone', 'Ciprofloxacin'];

export function PredictionWorkspace() {
  const [microbe, setMicrobe] = useState(microbes[0]);
  const [antibiotic, setAntibiotic] = useState(antibiotics[0]);
  const [result, setResult] = useState({
    probability: 0.73,
    confidence: 0.81,
    summary: 'High probability of resistance with carbapenem-sparing alternatives recommended.',
    alternatives: ['Nitrofurantoin', 'Fosfomycin', 'Piperacillin-Tazobactam'],
  });

  const runScenario = () => {
    const probability = Number((0.35 + Math.random() * 0.5).toFixed(2));
    setResult({
      probability,
      confidence: Number((0.55 + Math.random() * 0.35).toFixed(2)),
      summary: `${microbe} demonstrates an elevated resistance profile for ${antibiotic}; monitor MIC drift and confirm with AST before escalation.`,
      alternatives: ['Amikacin', 'Colistin', 'Ceftazidime-Avibactam'],
    });
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
      <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-6">
        <div className="mb-6 flex items-center gap-3">
          <Beaker className="h-5 w-5 text-cyan-300" />
          <div>
            <h3 className="text-xl font-semibold text-white">Prediction interface</h3>
            <p className="text-sm text-slate-400">Run clinician-facing AMR inference with explainability context.</p>
          </div>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          <label className="space-y-2 text-sm text-slate-300">
            <span>Microbe isolate</span>
            <select className="w-full rounded-2xl border border-white/10 bg-slate-900 px-4 py-3 text-white" value={microbe} onChange={(event) => setMicrobe(event.target.value)}>
              {microbes.map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </label>
          <label className="space-y-2 text-sm text-slate-300">
            <span>Antibiotic</span>
            <select className="w-full rounded-2xl border border-white/10 bg-slate-900 px-4 py-3 text-white" value={antibiotic} onChange={(event) => setAntibiotic(event.target.value)}>
              {antibiotics.map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </label>
        </div>
        <button className="mt-5 rounded-full bg-cyan-400 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300" onClick={runScenario} type="button">
          Generate AMR insight
        </button>
      </div>

      <motion.div
        initial={{ opacity: 0, x: 18 }}
        animate={{ opacity: 1, x: 0 }}
        className="rounded-3xl border border-cyan-400/20 bg-gradient-to-br from-cyan-500/10 via-slate-950/90 to-fuchsia-500/10 p-6"
      >
        <div className="mb-4 flex items-center gap-3">
          <ShieldCheck className="h-5 w-5 text-emerald-300" />
          <div>
            <h3 className="text-xl font-semibold text-white">AI recommendation summary</h3>
            <p className="text-sm text-slate-300">Probability, confidence, alternatives, and stewardship note.</p>
          </div>
        </div>
        <div className="space-y-4 text-sm text-slate-200">
          <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <p className="text-slate-400">Resistance probability</p>
            <p className="mt-2 text-3xl font-semibold text-white">{Math.round(result.probability * 100)}%</p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <p className="text-slate-400">Confidence score</p>
            <p className="mt-2 text-3xl font-semibold text-white">{Math.round(result.confidence * 100)}%</p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <p className="text-slate-400">Clinical insight</p>
            <p className="mt-2 leading-6">{result.summary}</p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <p className="text-slate-400">Alternative ranking</p>
            <div className="mt-3 flex flex-wrap gap-2">
              {result.alternatives.map((alternative) => (
                <span key={alternative} className="rounded-full border border-emerald-400/25 bg-emerald-400/10 px-3 py-1 text-emerald-200">
                  {alternative}
                </span>
              ))}
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
