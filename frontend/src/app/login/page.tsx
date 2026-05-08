import Link from 'next/link';

export default function LoginPage() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-[linear-gradient(180deg,_#020617,_#0f172a)] px-4 text-slate-100">
      <div className="w-full max-w-md rounded-[2rem] border border-white/10 bg-slate-950/75 p-8 shadow-2xl shadow-cyan-950/10">
        <p className="text-sm uppercase tracking-[0.35em] text-cyan-300">Secure access</p>
        <h1 className="mt-4 text-3xl font-semibold text-white">Welcome back</h1>
        <p className="mt-2 text-sm text-slate-400">Access AMR clinical intelligence, dashboards, and research operations.</p>
        <form className="mt-8 space-y-4">
          <label className="block text-sm text-slate-300">
            <span>Email</span>
            <input className="mt-2 w-full rounded-2xl border border-white/10 bg-slate-900 px-4 py-3 text-white" placeholder="doctor@hospital.org" type="email" />
          </label>
          <label className="block text-sm text-slate-300">
            <span>Password</span>
            <input className="mt-2 w-full rounded-2xl border border-white/10 bg-slate-900 px-4 py-3 text-white" placeholder="••••••••" type="password" />
          </label>
          <button className="w-full rounded-full bg-cyan-400 px-5 py-3 font-semibold text-slate-950 transition hover:bg-cyan-300" type="submit">
            Sign in
          </button>
        </form>
        <p className="mt-6 text-sm text-slate-400">
          Need an account?{' '}
          <Link className="text-cyan-300" href="/signup">
            Create one
          </Link>
        </p>
      </div>
    </main>
  );
}
