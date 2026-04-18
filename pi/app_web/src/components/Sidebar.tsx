import { Link, useLocation } from 'react-router-dom'
import { Shield, BarChart3, Activity, Settings, Users, Eye, Database, Monitor } from 'lucide-react'

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: BarChart3 },
  { name: 'Live Detection', href: '/live-detection', icon: Shield },
  { name: 'Model Comparison', href: '/model-comparison', icon: Activity },
  { name: 'Batch Analysis', href: '/batch-analysis', icon: Database },
  { name: 'Monitoring', href: '/monitoring', icon: Monitor },
  { name: 'Settings', href: '/settings', icon: Settings },
  { name: 'User Management', href: '/user-management', icon: Users },
]

export default function Sidebar() {
  const location = useLocation()

  return (
    <div className="w-64 bg-slate-900 min-h-screen">
      <div className="p-6">
        <div className="flex items-center gap-3">
          <Shield className="w-8 h-8 text-[#13739f]" />
          <h1 className="text-xl font-bold text-white">IOTinel</h1>
        </div>
        <p className="text-slate-400 text-sm mt-1">Smart City IDS</p>
      </div>

      <nav className="px-4 space-y-2">
        {navigation.map((item) => {
          const Icon = item.icon
          const isActive = location.pathname === item.href
          return (
            <Link
              key={item.name}
              to={item.href}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition ${
                isActive
                  ? 'bg-[#13739f] text-white'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span className="font-medium">{item.name}</span>
            </Link>
          )
        })}
      </nav>
    </div>
  )
}