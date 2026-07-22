const TOPIC_COLORS = {
  Pricing:  'bg-amber-500',
  UI:       'bg-fuchsia-500',
  Service:  'bg-cyan-500',
  Quality:  'bg-lime-500',
  Delivery: 'bg-sky-500',
  Support:  'bg-rose-500',
};

export default function TopicBars({ topics }) {
  return (
    <section className="rounded-2xl border border-slate-700/70 bg-slate-900/60 p-5 shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
          Business Topics
        </h2>
        <span className="text-xs text-slate-500">multi-label</span>
      </div>

      {topics.length === 0 ? (
        <p className="text-sm text-slate-500">No topics detected.</p>
      ) : (
        <ul className="space-y-3">
          {topics.map((t) => (
            <li key={t.label}>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-200 font-medium">{t.label}</span>
                <span className="text-slate-500">{(t.score * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                <div
                  className={`h-full ${TOPIC_COLORS[t.label] || 'bg-indigo-500'} transition-all duration-500`}
                  style={{ width: `${Math.max(4, t.score * 100)}%` }}
                />
              </div>
            </li>
          ))}
        </ul>
      )}

      <div className="mt-5 flex flex-wrap gap-2">
        {topics.map((t) => (
          <span
            key={`badge-${t.label}`}
            className="text-xs px-2.5 py-1 rounded-full border border-slate-700 text-slate-300"
          >
            #{t.label}
          </span>
        ))}
      </div>
    </section>
  );
}
