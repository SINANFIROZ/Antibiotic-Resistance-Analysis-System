const reports = [
  { patient: 'PT-10421', microbe: 'E. coli', antibiotic: 'Ciprofloxacin', status: 'Delivered', confidence: '84%' },
  { patient: 'PT-10422', microbe: 'K. pneumoniae', antibiotic: 'Ceftriaxone', status: 'Generated', confidence: '79%' },
  { patient: 'PT-10430', microbe: 'P. aeruginosa', antibiotic: 'Meropenem', status: 'Reviewed', confidence: '88%' },
];

export default function ReportsPage() {
  return (
    <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-6">
      <div className="mb-6">
        <p className="text-sm uppercase tracking-[0.2em] text-cyan-300">Historical reports</p>
        <h3 className="mt-2 text-2xl font-semibold text-white">Clinical export history</h3>
      </div>
      <div className="overflow-hidden rounded-3xl border border-white/10">
        <table className="min-w-full divide-y divide-white/10 text-left text-sm text-slate-300">
          <thead className="bg-white/5 text-xs uppercase tracking-[0.2em] text-slate-400">
            <tr>
              <th className="px-4 py-3">Patient</th>
              <th className="px-4 py-3">Microbe</th>
              <th className="px-4 py-3">Antibiotic</th>
              <th className="px-4 py-3">Confidence</th>
              <th className="px-4 py-3">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/10">
            {reports.map((report) => (
              <tr key={`${report.patient}-${report.antibiotic}`} className="bg-slate-950/40">
                <td className="px-4 py-4">{report.patient}</td>
                <td className="px-4 py-4">{report.microbe}</td>
                <td className="px-4 py-4">{report.antibiotic}</td>
                <td className="px-4 py-4">{report.confidence}</td>
                <td className="px-4 py-4">
                  <span className="rounded-full border border-emerald-400/25 bg-emerald-400/10 px-3 py-1 text-xs text-emerald-200">{report.status}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
