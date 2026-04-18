import { Routes, Route } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import LiveDetectionPage from './pages/LiveDetectionPage'
import ModelComparisonPage from './pages/ModelComparisonPage'
import BatchAnalysisPage from './pages/BatchAnalysisPage'
import MonitoringPage from './pages/MonitoringPage'
import SettingsPage from './pages/SettingsPage'
import UserManagementPage from './pages/UserManagementPage'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import Sidebar from './components/Sidebar'
import Header from './components/Header'

function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <div className="flex">
        <Sidebar />
        <div className="flex-1">
          <Header />
          <main className="p-6">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/live-detection" element={<LiveDetectionPage />} />
              <Route path="/model-comparison" element={<ModelComparisonPage />} />
              <Route path="/batch-analysis" element={<BatchAnalysisPage />} />
              <Route path="/monitoring" element={<MonitoringPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/user-management" element={<UserManagementPage />} />
              <Route path="/login" element={<LoginPage />} />
              <Route path="/register" element={<RegisterPage />} />
            </Routes>
          </main>
        </div>
      </div>
    </div>
  )
}

export default App