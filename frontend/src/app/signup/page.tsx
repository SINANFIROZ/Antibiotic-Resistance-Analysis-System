import Link from 'next/link';

export default function SignupPage() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-[linear-gradient(180deg,_#020617,_#0f172a)] px-4 text-slate-100">
      <div className="w-full max-w-xl rounded-[2rem] border border-white/10 bg-slate-950/75 p-8 shadow-2xl shadow-cyan-950/10">
        <p className="text-sm uppercase tracking-[0.35em] text-cyan-300">Platform onboarding</p>
        <h1 className="mt-4 text-3xl font-semibold text-white">Create your AMR workspace</h1>
        <form className="mt-8 grid gap-4 md:grid-cols-2">
          <label className="block text-sm text-slate-300">
            <span>Full name</span>
            <input className="mt-2 w-full rounded-2xl border border-white/10 bg-slate-900 px-4 py-3 text-white" placeholder="Dr. Jane Smith" type="text" />
          </label>
          <label className="block text-sm text-slate-300">
            <span>Role</span>
            <select className="mt-2 w-full rounded-2xl border border-white/10 bg-slate-900 px-4 py-3 text-white">
              <option>Doctor</option>
              <option>Researcher</option>
              <option>Lab Technician</option>
              <option>Admin</option>
            </select>
          </label>
          <label className="block text-sm text-slate-300 md:col-span-2">
            <span>Organization email</span>
            <input className="mt-2 w-full rounded-2xl border border-white/10 bg-slate-900 px-4 py-3 text-white" placeholder="research@institute.org" type="email" />
          </label>
          <label className="block text-sm text-slate-300 md:col-span-2">
            <span>Password</span>
            <input className="mt-2 w-full rounded-2xl border border-white/10 bg-slate-900 px-4 py-3 text-white" placeholder="Create a strong password" type="password" />
          </label>
          <button className="md:col-span-2 w-full rounded-full bg-cyan-400 px-5 py-3 font-semibold text-slate-950 transition hover:bg-cyan-300" type="submit">
            Create account
          </button>
        </form>
        <p className="mt-6 text-sm text-slate-400">
          Already onboarded?{' '}
          <Link className="text-cyan-300" href="/login">
            Sign in
          </Link>
        </p>
      </div>
    </main>
  );
}
