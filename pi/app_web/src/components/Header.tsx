import { Bell, User } from 'lucide-react'

export default function Header() {
  return (
    <header className="bg-slate-900 border-b border-slate-800 px-6 py-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white">Tableau de Bord</h2>
          <p className="text-slate-400 text-sm">Surveillance en temps réel des menaces IoT</p>
        </div>

        <div className="flex items-center gap-4">
          <button className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition">
            <Bell className="w-5 h-5 text-slate-300" />
          </button>

          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-[#13739f] flex items-center justify-center">
              <User className="w-4 h-4 text-white" />
            </div>
            <span className="text-sm text-slate-300">Admin</span>
          </div>
        </div>
      </div>
    </header>
  )
}