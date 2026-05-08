'use client';

import { create } from 'zustand';

type ThemeMode = 'light' | 'dark';

type UiStore = {
  theme: ThemeMode;
  activeRole: 'admin' | 'doctor' | 'researcher';
  setTheme: (theme: ThemeMode) => void;
  setActiveRole: (role: UiStore['activeRole']) => void;
};

export const useUiStore = create<UiStore>((set) => ({
  theme: 'dark',
  activeRole: 'doctor',
  setTheme: (theme) => set({ theme }),
  setActiveRole: (activeRole) => set({ activeRole }),
}));
