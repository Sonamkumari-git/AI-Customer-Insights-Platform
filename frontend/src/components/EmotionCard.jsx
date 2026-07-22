const EMOTION_STYLES = {
  Happy:      { badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40', bar: 'bg-emerald-500' },
  Angry:      { badge: 'bg-red-500/20 text-red-300 border-red-500/40',              bar: 'bg-red-500' },
  Sad:        { badge: 'bg-blue-500/20 text-blue-300 border-blue-500/40',           bar: 'bg-blue-500' },
  Frustrated: { badge: 'bg-orange-500/20 text-orange-300 border-orange-500/40',     bar: 'bg-orange-500' },
  Neutral:    { badge: 'bg-slate-500/20 text-slate-300 border-slate-500/40',        bar: 'bg-slate-400' },
};

export default function EmotionCard({ emotion }) {
  const style = EMOTION_STYLES[emotion.label] || EMOTION_STYLES.Neutral;
  const entries = Object.entries(emotion.distribution)
    .sort((a, b) => b[1] - a[1]);

  return (
    <section className="rounded-2xl border border-slate-700/70 bg-slate-900/60 p-5 shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
          Emotion
        </h2>
        <span className={`text-xs px-2.5 py-1 rounded-full border ${style.badge}`}>
          {emotion.label} · {(emotion.score * 100).toFixed(1)}%
        </span>
      </div>

      <ul className="space-y-3">
        {entries.map(([label, p]) => {
          const s = EMOTION_STYLES[label] || EMOTION_STYLES.Neutral;
          return (
            <li key={label}>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300">{label}</span>
                <span className="text-slate-500">{(p * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                <div
                  className={`h-full ${s.bar} transition-all duration-500`}
                  style={{ width: `${Math.max(2, p * 100)}%` }}
                />
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
