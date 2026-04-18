export default function LoginPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-950">
      <div className="rounded-3xl border border-slate-800 bg-white/5 p-8 w-full max-w-md">
        <h1 className="text-2xl font-semibold text-[#13739f] text-center mb-6">Connexion</h1>
        <form className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">Email</label>
            <input
              type="email"
              className="w-full rounded-xl border border-slate-700 bg-slate-900/50 px-3 py-2 text-white outline-none focus:border-[#13739f]"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">Mot de passe</label>
            <input
              type="password"
              className="w-full rounded-xl border border-slate-700 bg-slate-900/50 px-3 py-2 text-white outline-none focus:border-[#13739f]"
            />
          </div>
          <button
            type="submit"
            className="w-full rounded-xl bg-[#13739f] px-4 py-2 text-white font-semibold hover:bg-[#0f5a7a] transition"
          >
            Se connecter
          </button>
        </form>
      </div>
    </div>
  );
}