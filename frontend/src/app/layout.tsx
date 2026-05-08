import type { Metadata } from 'next';

import './globals.css';

export const metadata: Metadata = {
  title: 'AMR Intelligence Platform',
  description: 'Enterprise antimicrobial resistance intelligence platform for clinical, operational, and research workflows.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body>{children}</body>
    </html>
  );
}
