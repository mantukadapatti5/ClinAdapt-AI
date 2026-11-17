import { useEffect, useState } from "react";
type Finding = { label: string; score: number; uncertain: boolean };
type Ref = { source: string; text: string };
type Expl = { text: string; provider: string; note: string; references: Ref[] };
type GT = { label: string; value: number | null };
type Source = { dataset: string; dataset_name: string; image_id: string; findings: string[] };
type Result = { case_id: string | null; refused: boolean; quality: { status: string; issues: string[]; metrics: Record<string, number> };
  findings?: Finding[]; heatmap_png?: string; heatmap_label?: string; model_version?: string; trained?: boolean; explanation?: Expl; disclaimer: string;
  ground_truth?: GT[]; source?: Source | null };
type CaseRow = { id: string; created: number; model_version: string; source: string | null; last_action: string | null };
type GTCount = { positive: number; negative: number; uncertain: number; blank: number };
type DatasetSummary = { id: string; name: string; kind: "image" | "labels"; description: string; rows: number; images: number; available: boolean;
  mapped: Record<string, GTCount> | null; top_findings: { label: string; count: number }[] };
type DatasetStats = { id: string; rows: number; images: number; mapped: Record<string, GTCount> };
type DatasetItem = { id: string; findings: string[]; view: string; age: string; gender: string; ground_truth: GT[]; image_url: string; thumbnail_url: string };
type PreviewItem = { id: string; path: string; sex: string; age: string; view: string; ground_truth: GT[] };
const MODEL_LABELS = ["Pneumonia", "Pneumothorax", "Effusion"];
const MODES = [["concise", "Concise summary"], ["detailed", "Detailed explanation"], ["evidence", "Evidence-focused"]] as const;
const card = "rounded-xl bg-white p-5 shadow-sm ring-1 ring-slate-200";
const btn = "rounded-lg px-4 py-2 text-sm font-medium transition disabled:opacity-50";
const fmtGT = (v: number | null | undefined) => v === 1 ? "positive" : v === 0 ? "negative" : v === -1 ? "uncertain" : "unlabeled";
const gtClass = (v: number | null | undefined) => v === 1 ? "bg-rose-50 text-rose-700" : v === 0 ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-600";

export default function App() {
  const [file, setFile] = useState<File | null>(null); const [preview, setPreview] = useState("");
  const [fileSource, setFileSource] = useState<Source | null>(null);
  const [mode, setMode] = useState("concise"); const [provider, setProvider] = useState("local");
  const [res, setRes] = useState<Result | null>(null); const [expl, setExpl] = useState<Expl | null>(null);
  const [busy, setBusy] = useState(false); const [err, setErr] = useState(""); const [note, setNote] = useState("");
  const [msg, setMsg] = useState(""); const [cases, setCases] = useState<CaseRow[]>([]);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]); const [dataRoot, setDataRoot] = useState<string | null>(null);
  const [dsId, setDsId] = useState(""); const [dsFilter, setDsFilter] = useState("");
  const [dsStats, setDsStats] = useState<DatasetStats | null>(null); const [dsItems, setDsItems] = useState<DatasetItem[]>([]);
  const [dsTotal, setDsTotal] = useState(0); const [dsPreview, setDsPreview] = useState<PreviewItem[]>([]);
  const [dsNote, setDsNote] = useState(""); const [dsBusy, setDsBusy] = useState(false); const [selectedId, setSelectedId] = useState("");
  const loadCases = () => fetch("/api/cases").then(r => r.json()).then(setCases).catch(() => {});
  useEffect(() => { loadCases(); }, []);
  useEffect(() => {
    fetch("/api/datasets").then(r => r.json()).then(d => {
      setDatasets(d.datasets ?? []); setDataRoot(d.root ?? null);
      const first = (d.datasets ?? []).find((x: DatasetSummary) => x.kind === "image") ?? (d.datasets ?? [])[0];
      if (first) chooseDataset(first.id, first.kind);
    }).catch(() => setDsNote("Could not reach the dataset API."));
  }, []);

  async function chooseDataset(id: string, kind?: "image" | "labels") {
    const dsKind = kind ?? datasets.find(d => d.id === id)?.kind ?? "image";
    setDsId(id); setDsFilter(""); setDsStats(null); setDsItems([]); setDsPreview([]); setDsTotal(0); setDsNote(""); setSelectedId("");
    setDsBusy(true);
    try {
      const statsRes = await fetch(`/api/datasets/${id}/stats`); if (!statsRes.ok) throw new Error();
      setDsStats(await statsRes.json());
      const listRes = await fetch(dsKind === "image" ? `/api/datasets/${id}/items?limit=12` : `/api/datasets/${id}/preview?limit=8`);
      if (!listRes.ok) throw new Error();
      const list = await listRes.json(); setDsTotal(list.total);
      if (dsKind === "image") { setDsItems(list.items); setDsPreview([]); } else { setDsPreview(list.items); setDsItems([]); }
    } catch { setDsNote("Could not load this dataset. Check the backend log and data directory."); }
    finally { setDsBusy(false); }
  }
  async function applyFilter(finding: string) {
    setDsFilter(finding); setDsBusy(true); setSelectedId("");
    try {
      const r = await fetch(`/api/datasets/${dsId}/items?limit=12&finding=${encodeURIComponent(finding)}`);
      if (!r.ok) throw new Error(); const d = await r.json(); setDsItems(d.items); setDsTotal(d.total);
    } catch { setDsNote("Could not filter dataset images."); } finally { setDsBusy(false); }
  }
  async function loadMore() {
    setDsBusy(true);
    try {
      const r = await fetch(`/api/datasets/${dsId}/items?offset=${dsItems.length}&limit=12&finding=${encodeURIComponent(dsFilter)}`);
      if (!r.ok) throw new Error(); const d = await r.json();
      setDsItems(prev => [...prev, ...d.items]); setDsTotal(d.total);
    } catch { setDsNote("Could not load more images."); } finally { setDsBusy(false); }
  }

  async function analyzeFile(f: File, src: Source | null) {
    setBusy(true); setErr(""); setMsg("");
    const fd = new FormData();
    if (src) { fd.append("dataset", src.dataset); fd.append("image_id", src.image_id); } else fd.append("file", f);
    fd.append("mode", mode); fd.append("provider", provider);
    try {
      const r = await fetch("/api/analyze", { method: "POST", body: fd });
      if (!r.ok) throw new Error((await r.json()).detail ?? "Request failed");
      const d: Result = await r.json(); setRes(d); setExpl(d.explanation ?? null); loadCases();
    } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  }
  async function analyze() { if (file && !busy) await analyzeFile(file, fileSource); }
  async function useDatasetImage(item: DatasetItem) {
    const summary = datasets.find(d => d.id === dsId);
    const src: Source = { dataset: dsId, dataset_name: summary?.name ?? dsId, image_id: item.id, findings: item.findings };
    setSelectedId(item.id); setErr(""); setBusy(true);
    try {
      const r = await fetch(item.image_url); if (!r.ok) throw new Error("Could not load dataset image");
      const blob = await r.blob(); const f = new File([blob], item.id, { type: blob.type || "image/png" });
      setFile(f); setFileSource(src); setPreview(URL.createObjectURL(f)); setRes(null); setExpl(null);
      await analyzeFile(f, src);
    } catch (e) { setErr((e as Error).message); setBusy(false); }
  }
  async function reexplain(m: string, p: string) {  // predictions stay unchanged
    setMode(m); setProvider(p); if (!res?.case_id) return;
    const fd = new FormData(); fd.append("mode", m); fd.append("provider", p); setBusy(true);
    try { const r = await fetch(`/api/explain/${res.case_id}`, { method: "POST", body: fd }); setExpl(await r.json()); }
    finally { setBusy(false); }
  }
  async function review(action: string) {
    if (!res?.case_id) return;
    const fd = new FormData(); fd.append("case_id", res.case_id); fd.append("action", action); fd.append("mode", mode); fd.append("note", note);
    await fetch("/api/review", { method: "POST", body: fd }); setMsg(`Recorded: ${action}. Original AI output preserved.`); setNote(""); loadCases();
  }
  const pick = (f: File | null, src: Source | null = null) => {
    setFile(f); setFileSource(src); setRes(null); setExpl(null); setSelectedId(""); setPreview(f ? URL.createObjectURL(f) : "");
  };
  const current = datasets.find(d => d.id === dsId);
  const gtFor = (label: string) => res?.ground_truth?.find(g => g.label === label);

  const datasetCard = (
    <section className={`${card} space-y-3`}>
      <div className="flex items-center justify-between"><h2 className="font-semibold">Dataset browser</h2>{dsBusy && <span className="text-xs text-slate-400">loading…</span>}</div>
      <select value={dsId} onChange={e => chooseDataset(e.target.value)} className="w-full rounded-lg border border-slate-300 p-2 text-sm">
        {datasets.length === 0 && <option value="">No datasets found</option>}
        {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
      </select>
      {current && <p className="text-xs text-slate-500">{current.description}{dataRoot && <span className="ml-1 text-slate-400" title={dataRoot}>· {dataRoot}</span>}</p>}
      {dsStats && <div className="space-y-1 text-xs">
        <div className="flex flex-wrap gap-1">
          <span className="rounded bg-slate-100 px-1.5 py-0.5">{dsStats.rows.toLocaleString()} rows</span>
          {current?.kind === "image" && <span className="rounded bg-slate-100 px-1.5 py-0.5">{(dsStats.images ?? 0).toLocaleString()} images</span>}
          {MODEL_LABELS.map(l => dsStats.mapped[l] && <span key={l} className="rounded bg-rose-50 px-1.5 py-0.5 text-rose-700">{l} +{dsStats.mapped[l].positive.toLocaleString()}</span>)}
        </div>
        {current?.kind !== "image" && <p className="text-slate-400">CheXpert images are not bundled; label preview only.</p>}
      </div>}
      {current?.kind === "image" && <>
        <div className="flex flex-wrap gap-1">
          {["", ...MODEL_LABELS].map(f => <button key={f || "all"} onClick={() => applyFilter(f)} disabled={dsBusy}
            className={`rounded px-2 py-1 text-xs ${dsFilter === f ? "bg-sky-600 text-white" : "bg-slate-100 hover:bg-slate-200"}`}>{f || "All"}</button>)}
        </div>
        <div className="grid grid-cols-3 gap-2">
          {dsItems.map(item => <button key={item.id} onClick={() => useDatasetImage(item)} disabled={busy || dsBusy}
            title={`${item.id} · ${item.findings.join(", ") || "No Finding"}`}
            className={`rounded-lg p-1 text-left transition ${selectedId === item.id ? "ring-2 ring-sky-500" : "hover:bg-slate-50"}`}>
            <img src={item.thumbnail_url} alt={item.id} loading="lazy" className="aspect-square w-full rounded bg-black object-cover" />
            <span className="mt-1 block truncate text-[10px] text-slate-500">{item.id}</span>
          </button>)}
        </div>
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>Showing {dsItems.length} of {dsTotal.toLocaleString()}</span>
          {dsItems.length > 0 && dsItems.length < dsTotal && <button onClick={loadMore} disabled={dsBusy} className="rounded bg-slate-100 px-2 py-1 hover:bg-slate-200">Load more</button>}
        </div>
        <p className="text-[11px] text-slate-400">Click a thumbnail to analyse it against its dataset labels.</p>
      </>}
      {current?.kind === "labels" && dsPreview.length > 0 && <table className="w-full text-left text-xs">
        <thead className="text-slate-500"><tr><th className="pb-1">Path</th><th>Labels</th></tr></thead>
        <tbody>{dsPreview.map(row => <tr key={row.id} className="border-t align-top">
          <td className="max-w-[11rem] truncate py-1" title={row.path}>{row.id.split("/").slice(-3).join("/")}</td>
          <td className="py-1">{row.ground_truth.map(g => <span key={g.label} className={`mr-1 inline-block rounded px-1 ${gtClass(g.value)}`}>{g.label}: {fmtGT(g.value)}</span>)}</td>
        </tr>)}</tbody>
      </table>}
      {dsNote && <p className="rounded-lg bg-amber-50 p-2 text-xs text-amber-800">{dsNote}</p>}
    </section>
  );

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      <header className="bg-slate-900 px-6 py-4 text-white">
        <h1 className="text-xl font-semibold">ClinAdapt AI</h1>
        <p className="text-xs text-amber-300">Research and educational prototype. Not clinically validated. Not for diagnosis.</p>
      </header>
      <main className="mx-auto grid max-w-6xl gap-6 p-6 lg:grid-cols-3">
        <div className="space-y-6">
          <section className={`${card} space-y-4`}>
            <h2 className="font-semibold">1. Upload chest X-ray</h2>
            <input type="file" accept="image/png,image/jpeg" onChange={e => pick(e.target.files?.[0] ?? null)} className="block w-full text-sm file:mr-3 file:rounded-lg file:border-0 file:bg-sky-600 file:px-3 file:py-2 file:text-white" />
            {preview && <img src={preview} alt="Original X-ray" className="w-full rounded-lg bg-black" />}
            {fileSource && <p className="rounded-lg bg-sky-50 p-2 text-xs text-sky-900">Dataset image <b>{fileSource.image_id}</b> · {fileSource.dataset_name}<br />Labels: {fileSource.findings.join(", ") || "No Finding"}</p>}
            <p className="text-xs text-slate-500">Use only public, de-identified images. Images stay on your machine.</p>
            <div>
              <label className="text-sm font-medium">Explanation mode</label>
              <div className="mt-1 flex flex-col gap-1">{MODES.map(([k, l]) => (
                <button key={k} onClick={() => res ? reexplain(k, provider) : setMode(k)} className={`${btn} text-left ${mode === k ? "bg-sky-600 text-white" : "bg-slate-100 hover:bg-slate-200"}`}>{l}</button>))}</div>
            </div>
            <div>
              <label className="text-sm font-medium">GenAI provider</label>
              <div className="mt-1 flex gap-2">{[["local", "Local (Ollama)"], ["external", "External API"]].map(([k, l]) => (
                <button key={k} onClick={() => res ? reexplain(mode, k) : setProvider(k)} className={`${btn} flex-1 ${provider === k ? "bg-sky-600 text-white" : "bg-slate-100 hover:bg-slate-200"}`}>{l}</button>))}</div>
            </div>
            <button onClick={analyze} disabled={!file || busy} className={`${btn} w-full bg-emerald-600 text-white hover:bg-emerald-700`}>{busy ? "Working..." : "Analyse"}</button>
            {err && <p className="rounded-lg bg-red-50 p-2 text-sm text-red-700">{err}</p>}
          </section>
          {datasetCard}
        </div>

        <section className="space-y-6 lg:col-span-2">
          {!res && <div className={`${card} text-sm text-slate-500`}>Upload an image or pick a dataset thumbnail and press Analyse. Results, Grad-CAM, uncertainty and explanations appear here.</div>}
          {res?.refused && <div className={`${card} text-amber-800`}>Image flagged and not analysed: {res.quality.issues.join(", ")}.{res.source && ` Source: ${res.source.image_id}.`}</div>}
          {res && !res.refused && res.findings && (<>
            {res.source && <div className="rounded-xl bg-sky-50 p-3 text-xs text-sky-900 ring-1 ring-sky-200">
              <b>{res.source.image_id}</b> from {res.source.dataset_name}. Labeled findings: {res.source.findings.join(", ") || "No Finding"}. Dataset labels are research annotations, not confirmed diagnoses.
            </div>}
            {!res.trained && <div className="rounded-xl bg-red-50 p-4 text-sm text-red-800 ring-1 ring-red-200"><b>Untrained demo model.</b> Scores are meaningless until you train and add weights (see README).</div>}
            <div className={card}>
              <h2 className="mb-3 font-semibold">2. Results <span className="text-xs font-normal text-slate-400">{res.case_id} · {res.model_version}</span></h2>
              <div className="grid gap-4 md:grid-cols-2">
                <div>{res.findings.map(f => {
                  const gt = gtFor(f.label)?.value;
                  const agrees = gt === 1 || gt === 0 ? (f.score >= 0.5) === (gt === 1) : null;
                  return <div key={f.label} className="mb-3">
                    <div className="flex justify-between text-sm"><span>{f.label === "Effusion" ? "Effusion-related finding" : f.label}</span>
                      <span>{(f.score * 100).toFixed(1)}%{f.uncertain && <span className="ml-2 rounded bg-amber-100 px-1.5 text-xs text-amber-800">uncertain</span>}</span></div>
                    <div className="h-2 rounded bg-slate-200"><div className="h-2 rounded bg-sky-600" style={{ width: `${f.score * 100}%` }} /></div>
                    {gt !== undefined && <p className="mt-1 text-xs">
                      <span className={`rounded px-1.5 py-0.5 ${gtClass(gt)}`}>dataset label: {fmtGT(gt)}</span>
                      {agrees !== null && <span className={`ml-1 rounded px-1.5 py-0.5 ${agrees ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}`}>{agrees ? "model agrees" : "model differs"}</span>}
                    </p>}
                  </div>;
                })}
                  <p className={`mt-2 rounded-lg p-2 text-sm ${res.quality.status === "suitable" ? "bg-emerald-50 text-emerald-800" : "bg-amber-50 text-amber-800"}`}>
                    Technical quality: {res.quality.status}{res.quality.issues.length ? ` (${res.quality.issues.join(", ")})` : ""}</p>
                </div>
                <div><img src={`data:image/png;base64,${res.heatmap_png}`} alt="Grad-CAM overlay" className="w-full rounded-lg" />
                  <p className="mt-1 text-xs text-slate-500">Grad-CAM for “{res.heatmap_label}”: model attention, not proof of disease.</p></div>
              </div>
            </div>
            {expl && <div className={card}>
              <h2 className="mb-2 font-semibold">3. Explanation <span className="text-xs font-normal text-slate-400">via {expl.provider}</span></h2>
              {expl.note && <p className="mb-2 text-xs text-amber-700">{expl.note}</p>}
              <p className="whitespace-pre-wrap text-sm leading-relaxed">{expl.text}</p>
              <ul className="mt-3 space-y-1 border-t pt-3 text-xs text-slate-500">{expl.references.map((r, i) => <li key={i}>[{i + 1}] <b>{r.source}</b>: {r.text}</li>)}</ul>
            </div>}
            <div className={card}>
              <h2 className="mb-2 font-semibold">4. Doctor review</h2>
              <textarea value={note} onChange={e => setNote(e.target.value)} placeholder="Note / modification / question" className="mb-3 w-full rounded-lg border border-slate-300 p-2 text-sm" rows={2} />
              <div className="flex flex-wrap gap-2">{[["accept", "bg-emerald-600"], ["modify", "bg-sky-600"], ["question", "bg-amber-500"], ["reject", "bg-red-600"]].map(([a, c]) => (
                <button key={a} onClick={() => review(a)} className={`${btn} ${c} capitalize text-white`}>{a}</button>))}</div>
              {msg && <p className="mt-2 text-sm text-emerald-700">{msg}</p>}
            </div>
          </>)}
          <div className={card}>
            <h2 className="mb-2 font-semibold">Case history</h2>
            {cases.length === 0 ? <p className="text-sm text-slate-500">No cases yet.</p> :
              <table className="w-full text-left text-sm"><thead className="text-xs text-slate-500"><tr><th>Case</th><th>Time</th><th>Model</th><th>Source</th><th>Last review</th></tr></thead>
                <tbody>{cases.map(c => <tr key={c.id} className="border-t"><td>{c.id}</td><td>{new Date(c.created * 1000).toLocaleString()}</td><td>{c.model_version}</td>
                  <td className="max-w-[10rem] truncate text-xs" title={c.source ?? "upload"}>{c.source ?? "upload"}</td><td>{c.last_action ?? "pending"}</td></tr>)}</tbody></table>}
          </div>
        </section>
      </main>
    </div>
  );
}
