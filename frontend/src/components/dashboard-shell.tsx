'use client';

import Link from 'next/link';
import { Activity, BarChart3, FileText, FlaskConical, LayoutDashboard, Shield } from 'lucide-react';
import { usePathname } from 'next/navigation';

import { ThemeToggle } from '@/components/theme-toggle';
import { cn } from '@/lib/utils';

const navItems = [
  { href: '/dashboard/doctor', label: 'Doctor', icon: Activity },
  { href: '/dashboard/admin', label: 'Admin', icon: Shield },
  { href: '/dashboard/research', label: 'Research', icon: FlaskConical },
  { href: '/dashboard/predictions', label: 'Predictions', icon: LayoutDashboard },
  { href: '/dashboard/reports', label: 'Reports', icon: FileText },
];

export function DashboardShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(34,211,238,0.18),_transparent_35%),radial-gradient(circle_at_right,_rgba(168,85,247,0.18),_transparent_30%),linear-gradient(180deg,_#020617,_#0f172a)] text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col gap-6 px-4 py-6 lg:flex-row lg:px-6">
        <aside className="w-full rounded-3xl border border-white/10 bg-slate-950/70 p-5 lg:w-72">
          <div className="mb-8">
            <p className="text-xs uppercase tracking-[0.35em] text-cyan-300">AMR SaaS</p>
            <h1 className="mt-3 text-2xl font-semibold text-white">Clinical intelligence</h1>
            <p className="mt-2 text-sm text-slate-400">Decision support for doctors, labs, and researchers.</p>
          </div>
          <nav className="space-y-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    'flex items-center gap-3 rounded-2xl px-4 py-3 text-sm transition',
                    pathname === item.href ? 'bg-cyan-400 text-slate-950' : 'text-slate-300 hover:bg-white/5 hover:text-white',
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </Link>
              );
            })}
          </nav>
          <div className="mt-8 rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-slate-300">
            <p className="font-medium text-white">Model governance</p>
            <p className="mt-2">Explainable AI, audit logging, and report traceability are enabled at the platform layer.</p>
          </div>
        </aside>
        <main className="flex-1">
          <div className="mb-6 flex flex-col gap-4 rounded-3xl border border-white/10 bg-slate-950/70 p-5 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-cyan-300">Operational dashboard</p>
              <h2 className="mt-2 text-3xl font-semibold text-white">AMR intelligence workspace</h2>
            </div>
            <div className="flex items-center gap-3">
              <button className="rounded-full border border-white/10 px-4 py-2 text-sm text-slate-200">Export PDF</button>
              <ThemeToggle />
            </div>
          </div>
          {children}
        </main>
      </div>
    </div>
  );
}
