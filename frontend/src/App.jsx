import { useEffect, useState } from 'react'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

const MODELS = [
  ['logistic_regression', 'Logistic Regression'],
  ['passive_aggressive', 'Passive-Aggressive'],
  ['naive_bayes', 'Multinomial Naive Bayes'],
  ['all', 'Compare all three'],
]

const MODEL_NAMES = {
  logistic_regression: 'Logistic Regression',
  passive_aggressive: 'Passive-Aggressive',
  naive_bayes: 'Multinomial Naive Bayes',
}

const percent = value => `${(value * 100).toFixed(1)}%`

const verdictTone = {
  FAKE: { text: 'text-flagged', border: 'border-flagged', bg: 'bg-flagged/10' },
  REAL: { text: 'text-verified', border: 'border-verified', bg: 'bg-verified/10' },
  UNCERTAIN: { text: 'text-pending', border: 'border-pending', bg: 'bg-pending/10' },
}

function Stat({ value, label }) {
  return (
    <div>
      <p className="font-serif text-3xl text-paper">{value}</p>
      <p className="mt-1 text-xs text-muted">{label}</p>
    </div>
  )
}

function EvaluationDashboard({ metrics, error, loading }) {
  if (loading) {
    return <p className="mt-10 text-sm text-muted">Loading saved evaluation metrics…</p>
  }

  if (error) {
    return (
      <div role="alert" className="mt-10 border-l-2 border-flagged py-3 pl-4 text-sm text-flagged">
        <p>{error}</p>
        <p className="mt-1 text-muted">
          Confirm the backend is running and outputs/classical_metrics.json exists.
        </p>
      </div>
    )
  }

  if (!metrics) return null

  const modelEntries = Object.entries(metrics.models || {})

  return (
    <section className="mt-16 border-t border-rule pt-10">
      <h2 className="font-serif text-2xl text-paper">How the classifiers were evaluated</h2>
      <p className="mt-2 max-w-[62ch] text-sm leading-6 text-muted">
        Figures below come from the saved test-set run, not from anything you just typed above.
        They describe performance on held-out data — not whether any single article is true.
      </p>

      <div className="mt-8 flex flex-wrap gap-x-10 gap-y-6 border-y border-rule py-6">
        <Stat value={Number(metrics.train_rows).toLocaleString()} label="training records" />
        <Stat value={Number(metrics.test_rows).toLocaleString()} label="held-out test records" />
        <Stat value={Number(metrics.rows_after_deduplication).toLocaleString()} label="records after dedup" />
        <Stat value={metrics.seed} label="random seed" />
      </div>

      <div className="mt-10">
        <h3 className="font-serif text-lg text-paper">Classifier comparison</h3>
        <div className="mt-4 divide-y divide-rule">
          {modelEntries.map(([id, result]) => (
            <div key={id} className="grid grid-cols-[1fr_auto] items-center gap-x-6 gap-y-2 py-4 sm:grid-cols-[180px_1fr_auto]">
              <span className="text-sm text-paper">{MODEL_NAMES[id] || id}</span>

              <div
                className="hidden h-1.5 overflow-hidden rounded-full bg-rule sm:block"
                role="img"
                aria-label={`${MODEL_NAMES[id] || id} accuracy ${percent(result.accuracy)}`}
              >
                <div
                  className="h-full bg-verified"
                  style={{ width: `${Math.max(0, Math.min(100, result.accuracy * 100))}%` }}
                />
              </div>

              <span className="justify-self-end font-serif text-lg text-paper">
                {percent(result.accuracy)}
              </span>

              <div className="col-span-2 flex gap-5 text-xs text-muted sm:col-span-3">
                <span>Precision {percent(result.macro_precision)}</span>
                <span>Recall {percent(result.macro_recall)}</span>
                <span>Macro F1 {percent(result.macro_f1)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-12">
        <h3 className="font-serif text-lg text-paper">Confusion matrices</h3>
        <p className="mt-2 text-sm text-muted">
          Rows are actual labels, columns are predicted labels — in FAKE, then REAL order.
        </p>

        <div className="mt-5 grid gap-8 lg:grid-cols-3">
          {modelEntries.map(([id, result]) => {
            const matrix = result.confusion_matrix_fake_real
            return (
              <div key={id}>
                <h4 className="mb-3 text-sm text-paper">{MODEL_NAMES[id] || id}</h4>
                {Array.isArray(matrix) && matrix.length === 2 ? (
                  <table className="w-full border-collapse text-sm">
                    <thead>
                      <tr className="text-muted">
                        <th className="pb-2 text-left font-normal">Actual \ Pred.</th>
                        <th className="pb-2 font-normal">FAKE</th>
                        <th className="pb-2 font-normal">REAL</th>
                      </tr>
                    </thead>
                    <tbody>
                      {matrix.map((row, index) => (
                        <tr key={index} className="border-t border-rule">
                          <th className="py-2 text-left font-normal text-muted">
                            {index === 0 ? 'FAKE' : 'REAL'}
                          </th>
                          {row.map((count, column) => (
                            <td key={column} className="py-2 text-center font-serif text-paper">
                              {Number(count).toLocaleString()}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p className="text-sm text-muted">Unavailable.</p>
                )}
              </div>
            )
          })}
        </div>
      </div>

      <div className="mt-10 max-w-[62ch] text-sm leading-6 text-muted">
        <p><span className="text-paper">Dataset:</span> {metrics.dataset}</p>
        <p className="mt-1"><span className="text-paper">Label mapping:</span> 0 = FAKE, 1 = REAL</p>
        <p className="mt-3">
          These results are specific to this dataset, preprocessing, split, and training run.
          They're not a guarantee of performance on news from other sources.
        </p>
      </div>
    </section>
  )
}

export default function App() {
  const [text, setText] = useState('')
  const [model, setModel] = useState('all')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [health, setHealth] = useState('checking')

  const [metrics, setMetrics] = useState(null)
  const [metricsError, setMetricsError] = useState('')
  const [metricsLoading, setMetricsLoading] = useState(true)

  useEffect(() => {
    let cancelled = false

    async function loadDashboard() {
      try {
        const response = await fetch(`${API}/metrics`)
        if (!response.ok) throw new Error(`Metrics request failed (${response.status})`)
        const data = await response.json()
        if (!cancelled) setMetrics(data)
      } catch (err) {
        if (!cancelled) setMetricsError(err.message)
      } finally {
        if (!cancelled) setMetricsLoading(false)
      }
    }

    async function checkHealth() {
      try {
        const response = await fetch(`${API}/health`)
        const data = await response.json()
        if (!cancelled) setHealth(data.models_trained ? 'ready' : 'not-trained')
      } catch {
        if (!cancelled) setHealth('offline')
      }
    }

    loadDashboard()
    checkHealth()
    return () => { cancelled = true }
  }, [])

  async function analyze() {
    setError('')
    setResult(null)

    if (text.trim().length < 10) {
      setError('Enter at least 10 characters.')
      return
    }

    setLoading(true)
    try {
      const response = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, model }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Request failed')
      setResult(data)
    } catch (err) {
      setError(`${err.message}. Check that the backend is running and trained.`)
    } finally {
      setLoading(false)
    }
  }

  const tone = result ? (verdictTone[result.overall] || verdictTone.UNCERTAIN) : null

  return (
    <div className="min-h-screen bg-ink font-sans text-paper">
      <header className="flex items-center justify-between px-6 py-6 sm:px-10">
        <div className="flex items-baseline gap-3">
          <span className="font-serif text-xl">NEWSCHECK</span>
          <span className="text-xs text-muted">text-pattern classifier, not a fact-checker</span>
        </div>
        <span className="flex items-center gap-2 text-xs text-muted">
          <span className={`h-1.5 w-1.5 rounded-full ${
            health === 'ready' ? 'bg-verified' : health === 'checking' ? 'bg-pending' : 'bg-flagged'
          }`} />
          {health}
        </span>
      </header>

      <main className="px-6 pb-20 sm:px-10">
        <div className="mx-auto max-w-3xl">
          <h1 className="mt-6 max-w-[20ch] font-serif text-4xl leading-tight sm:text-5xl">
            What does the language of this article actually look like?
          </h1>
          <p className="mt-4 max-w-[58ch] text-sm leading-6 text-muted">
            Paste an article below. Three classifiers, each trained on the same labeled dataset,
            will separately judge the text and report how confident each one is.
          </p>

          <section className="mt-10">
            <label htmlFor="news" className="text-sm text-muted">Article text</label>
            <textarea
              id="news"
              value={text}
              onChange={event => setText(event.target.value)}
              maxLength={20000}
              className="mt-2 min-h-56 w-full rounded-sm border border-rule bg-paper p-4 text-ink outline-none placeholder:text-ink/40 focus:border-verified"
              placeholder="Paste a headline or article…"
            />
            <div className="mt-2 flex items-center justify-between text-xs text-muted">
              <span>{text.length.toLocaleString()} / 20,000</span>
            </div>

            <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center">
              <select
                id="model"
                value={model}
                onChange={event => setModel(event.target.value)}
                className="rounded-sm border border-rule bg-ink p-3 text-sm sm:flex-1"
              >
                {MODELS.map(([id, name]) => (
                  <option key={id} value={id}>{name}</option>
                ))}
              </select>
              <button
                onClick={analyze}
                disabled={loading}
                className="rounded-sm bg-verified px-6 py-3 text-sm font-medium text-ink disabled:opacity-50"
              >
                {loading ? 'Reading…' : 'Analyze'}
              </button>
            </div>
            {error && <p role="alert" className="mt-3 text-sm text-flagged">{error}</p>}
          </section>

          <section className="mt-14">
            {!result ? (
              <p className="text-sm text-muted">Your result will appear here — nothing analyzed yet.</p>
            ) : (
              <div className="flex flex-col gap-8 sm:flex-row sm:items-start">
                <div
                  className={`inline-block shrink-0 -rotate-2 border-4 border-double px-6 py-3 ${tone.border} ${tone.bg}`}
                >
                  <span className={`font-serif text-3xl tracking-wide ${tone.text}`}>
                    {result.overall}
                  </span>
                </div>

                <div className="flex-1">
                  <p className="text-xs text-muted">{result.model_count} model result(s)</p>
                  <div className="mt-3 divide-y divide-rule">
                    {result.results.map(item => (
                      <div key={item.model} className="flex items-center justify-between gap-4 py-2.5 text-sm">
                        <span>{MODEL_NAMES[item.model] || item.model}</span>
                        <span className="flex items-center gap-3 text-muted">
                          <span className={verdictTone[item.label]?.text || 'text-paper'}>
                            {item.label}
                          </span>
                          <span>
                            {item.confidence == null ? 'no estimate' : `${(item.confidence * 100).toFixed(1)}%`}
                          </span>
                        </span>
                      </div>
                    ))}
                  </div>
                  <p className="mt-4 max-w-[52ch] text-xs leading-5 text-muted">{result.warning}</p>
                </div>
              </div>
            )}
          </section>

          <EvaluationDashboard metrics={metrics} error={metricsError} loading={metricsLoading} />

          <footer className="mt-16 border-t border-rule pt-6 text-xs text-muted">
            Evaluation results are loaded from the saved training run. Predictions are text
            classifications, not independent fact-checks.
          </footer>
        </div>
      </main>
    </div>
  )
}
