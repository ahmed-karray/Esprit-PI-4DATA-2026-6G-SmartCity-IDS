import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const shapData = [
  { feature: 'Dur', importance: 0.85 },
  { feature: 'TotPkts', importance: 0.72 },
  { feature: 'Rate', importance: 0.68 },
  { feature: 'Load', importance: 0.61 },
  { feature: 'Loss', importance: 0.45 },
  { feature: 'TcpRtt', importance: 0.38 },
];

export default function LiveDetectionPage() {
  const [inputData, setInputData] = useState({
    Dur: '',
    TotPkts: '',
    TotBytes: '',
    Rate: '',
    Load: '',
    Loss: '',
    pLoss: '',
    TcpRtt: '',
  });
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [showShap, setShowShap] = useState(false);

  const handleInputChange = (field: string, value: string) => {
    setInputData(prev => ({ ...prev, [field]: value }));
  };

  const handleDetect = () => {
    // Mock detection logic
    const isAttack = Math.random() > 0.7;
    setPrediction(isAttack ? 'Attack Detected' : 'Normal Traffic');
    setConfidence(Math.random() * 0.3 + 0.7); // 70-100%
    setShowShap(true);
  };

  const handleReset = () => {
    setInputData({
      Dur: '',
      TotPkts: '',
      TotBytes: '',
      Rate: '',
      Load: '',
      Loss: '',
      pLoss: '',
      TcpRtt: '',
    });
    setPrediction(null);
    setConfidence(null);
    setShowShap(false);
  };

  const handleGenerateData = () => {
    // Generate realistic IoT network data based on typical ranges
    const generatedData = {
      Dur: (Math.random() * 100 + 1).toFixed(2), // 1-101 seconds
      TotPkts: Math.floor(Math.random() * 10000 + 100).toString(), // 100-10100 packets
      TotBytes: Math.floor(Math.random() * 1000000 + 10000).toString(), // 10KB-1MB
      Rate: (Math.random() * 1000 + 10).toFixed(2), // 10-1010 packets/sec
      Load: (Math.random() * 100 + 1).toFixed(2), // 1-101% load
      Loss: Math.floor(Math.random() * 100 + 1).toString(), // 1-101 lost packets
      pLoss: (Math.random() * 10).toFixed(2), // 0-10% packet loss
      TcpRtt: (Math.random() * 500 + 10).toFixed(2), // 10-510ms RTT
    };
    setInputData(generatedData);
    setPrediction(null);
    setConfidence(null);
    setShowShap(false);
  };

  return (
    <div className="space-y-6">
      <div className="rounded-3xl border border-slate-800 bg-white/5 p-6 shadow-soft">
        <h1 className="text-2xl font-semibold text-[#13739f]">Live Detection</h1>
        <p className="mt-2 text-slate-400">Exécution de détection en temps réel et explications SHAP.</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Input Form */}
        <div className="rounded-3xl border border-slate-800 bg-white/5 p-6">
          <h3 className="text-lg font-semibold text-[#13739f] mb-4">Paramètres de Détection</h3>
          <div className="grid gap-4 sm:grid-cols-2">
            {Object.entries(inputData).map(([key, value]) => (
              <div key={key}>
                <label className="block text-sm font-medium text-slate-300 mb-1">{key}</label>
                <input
                  type="number"
                  value={value}
                  onChange={(e) => handleInputChange(key, e.target.value)}
                  className="w-full rounded-xl border border-slate-700 bg-slate-900/50 px-3 py-2 text-sm text-white outline-none focus:border-[#13739f]"
                  placeholder="0.0"
                />
              </div>
            ))}
          </div>
          <div className="flex gap-3 mt-6">
            <button
              onClick={handleGenerateData}
              className="flex-1 rounded-xl bg-green-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-green-700"
            >
              Générer des Données
            </button>
            <button
              onClick={handleDetect}
              className="flex-1 rounded-xl bg-[#13739f] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[#0f5a7a]"
            >
              Détecter
            </button>
            <button
              onClick={handleReset}
              className="rounded-xl border border-slate-700 bg-slate-900/50 px-4 py-2 text-sm font-semibold text-slate-300 transition hover:bg-slate-800"
            >
              Réinitialiser
            </button>
          </div>
        </div>

        {/* Results */}
        <div className="rounded-3xl border border-slate-800 bg-white/5 p-6">
          <h3 className="text-lg font-semibold text-[#13739f] mb-4">Résultats</h3>
          {prediction ? (
            <div className="space-y-4">
              <div className={`rounded-xl p-4 ${prediction.includes('Attack') ? 'bg-red-500/20 border border-red-500/50' : 'bg-green-500/20 border border-green-500/50'}`}>
                <p className="text-lg font-semibold">{prediction}</p>
                {confidence && (
                  <p className="text-sm text-slate-300 mt-1">
                    Confiance: {(confidence * 100).toFixed(1)}%
                  </p>
                )}
              </div>
              <button
                onClick={() => setShowShap(!showShap)}
                className="w-full rounded-xl bg-slate-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-600"
              >
                {showShap ? 'Masquer' : 'Afficher'} Explication SHAP
              </button>
            </div>
          ) : (
            <p className="text-slate-400">Générez des données ou entrez les paramètres et cliquez sur "Détecter" pour obtenir un résultat.</p>
          )}
        </div>
      </div>

      {/* SHAP Explanation */}
      {showShap && (
        <div className="rounded-3xl border border-slate-800 bg-white/5 p-6">
          <h3 className="text-lg font-semibold text-[#13739f] mb-4">Explication SHAP - Importance des Fonctionnalités</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={shapData} layout="horizontal">
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis type="number" stroke="#9ca3af" />
              <YAxis dataKey="feature" type="category" stroke="#9ca3af" width={80} />
              <Tooltip />
              <Bar dataKey="importance" fill="#13739f" />
            </BarChart>
          </ResponsiveContainer>
          <p className="text-sm text-slate-400 mt-4">
            Cette visualisation montre l'importance relative de chaque fonctionnalité dans la décision du modèle.
          </p>
        </div>
      )}
    </div>
  );
}