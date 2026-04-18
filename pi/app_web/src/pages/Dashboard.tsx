import { Shield, AlertTriangle, Activity, TrendingUp } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts'

const threatData = [
  { time: '00:00', threats: 12 },
  { time: '04:00', threats: 8 },
  { time: '08:00', threats: 25 },
  { time: '12:00', threats: 35 },
  { time: '16:00', threats: 28 },
  { time: '20:00', threats: 18 },
]

const deviceData = [
  { device: 'Smart Meter', attacks: 45 },
  { device: 'Traffic Light', attacks: 32 },
  { device: 'Security Cam', attacks: 28 },
  { device: 'Street Light', attacks: 15 },
  { device: 'Parking Sensor', attacks: 12 },
]

export default function Dashboard() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="rounded-3xl border border-slate-800 bg-white/5 p-6 shadow-soft">
        <h1 className="text-2xl font-semibold text-[#13739f]">Tableau de Bord</h1>
        <p className="mt-2 text-slate-400">Vue d'ensemble de la sécurité IoT en temps réel</p>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-3xl border border-slate-800 bg-white/5 p-6">
          <div className="flex items-center gap-3">
            <Shield className="w-8 h-8 text-green-400" />
            <div>
              <p className="text-sm text-slate-400">Détections</p>
              <p className="text-2xl font-bold text-white">1,247</p>
            </div>
          </div>
        </div>

        <div className="rounded-3xl border border-slate-800 bg-white/5 p-6">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-8 h-8 text-red-400" />
            <div>
              <p className="text-sm text-slate-400">Alertes Actives</p>
              <p className="text-2xl font-bold text-white">23</p>
            </div>
          </div>
        </div>

        <div className="rounded-3xl border border-slate-800 bg-white/5 p-6">
          <div className="flex items-center gap-3">
            <Activity className="w-8 h-8 text-blue-400" />
            <div>
              <p className="text-sm text-slate-400">Appareils Surveillés</p>
              <p className="text-2xl font-bold text-white">8,542</p>
            </div>
          </div>
        </div>

        <div className="rounded-3xl border border-slate-800 bg-white/5 p-6">
          <div className="flex items-center gap-3">
            <TrendingUp className="w-8 h-8 text-purple-400" />
            <div>
              <p className="text-sm text-slate-400">Taux de Précision</p>
              <p className="text-2xl font-bold text-white">94.2%</p>
            </div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Threat Timeline */}
        <div className="rounded-3xl border border-slate-800 bg-white/5 p-6">
          <h3 className="text-lg font-semibold text-[#13739f] mb-4">Évolution des Menaces (24h)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={threatData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip />
              <Line type="monotone" dataKey="threats" stroke="#13739f" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Device Attacks */}
        <div className="rounded-3xl border border-slate-800 bg-white/5 p-6">
          <h3 className="text-lg font-semibold text-[#13739f] mb-4">Attaques par Type d'Appareil</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={deviceData} layout="horizontal">
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis type="number" stroke="#9ca3af" />
              <YAxis dataKey="device" type="category" stroke="#9ca3af" width={100} />
              <Tooltip />
              <Bar dataKey="attacks" fill="#13739f" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Alerts */}
      <div className="rounded-3xl border border-slate-800 bg-white/5 p-6">
        <h3 className="text-lg font-semibold text-[#13739f] mb-4">Alertes Récentes</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between p-4 rounded-xl bg-red-500/10 border border-red-500/20">
            <div>
              <p className="font-semibold text-red-400">Attaque DDoS Détectée</p>
              <p className="text-sm text-slate-400">Smart Meter Cluster - Secteur Nord</p>
            </div>
            <span className="text-sm text-slate-500">2 min ago</span>
          </div>

          <div className="flex items-center justify-between p-4 rounded-xl bg-yellow-500/10 border border-yellow-500/20">
            <div>
              <p className="font-semibold text-yellow-400">Comportement Anormal</p>
              <p className="text-sm text-slate-400">Traffic Light Controller - Intersection Centrale</p>
            </div>
            <span className="text-sm text-slate-500">15 min ago</span>
          </div>

          <div className="flex items-center justify-between p-4 rounded-xl bg-blue-500/10 border border-blue-500/20">
            <div>
              <p className="font-semibold text-blue-400">Scan de Port Détecté</p>
              <p className="text-sm text-slate-400">Security Camera Network - Zone Résidentielle</p>
            </div>
            <span className="text-sm text-slate-500">1h ago</span>
          </div>
        </div>
      </div>
    </div>
  )
}