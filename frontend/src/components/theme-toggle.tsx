'use client';

import { MoonStar, SunMedium } from 'lucide-react';
import { useEffect } from 'react';

import { useUiStore } from '@/store/ui-store';

export function ThemeToggle() {
  const { theme, setTheme } = useUiStore();

  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle('dark', theme === 'dark');
    window.localStorage.setItem('amr-theme', theme);
  }, [theme]);

  useEffect(() => {
    const savedTheme = window.localStorage.getItem('amr-theme');
    if (savedTheme === 'light' || savedTheme === 'dark') {
      setTheme(savedTheme);
    }
  }, [setTheme]);

  return (
    <button
      aria-label="Toggle theme"
      className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 transition hover:border-cyan-400/50 hover:bg-cyan-500/10 dark:text-slate-100"
      onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
      type="button"
    >
      {theme === 'dark' ? <SunMedium className="h-4 w-4" /> : <MoonStar className="h-4 w-4" />}
      {theme === 'dark' ? 'Light mode' : 'Dark mode'}
    </button>
  );
}
