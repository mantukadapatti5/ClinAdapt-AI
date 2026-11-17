import io, json, time, uuid
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from . import datasets, db, llm, model, quality
app = FastAPI(title="ClinAdapt AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
DISCLAIMER = "Research and educational prototype. Not clinically validated. Not for diagnosis."

def _dataset_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except datasets.DatasetNotAvailable as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/api/health")
def health():
    paths = datasets.data_paths()
    return {"ok": True, "model": model.VERSION, "disclaimer": DISCLAIMER,
            "datasets_root": str(paths["root"]) if paths["root"] else None,
            "nih_images": bool(paths["nih_images"]), "chexpert_labels": bool(paths["chexbert"] or paths["visualchexbert"])}

@app.get("/api/datasets")
def dataset_list(): return _dataset_call(datasets.list_datasets)

@app.get("/api/datasets/verify")
async def dataset_verify(): return await run_in_threadpool(_dataset_call, datasets.verify)

@app.get("/api/datasets/{dataset_id}/stats")
def dataset_stats(dataset_id: str): return _dataset_call(datasets.stats, dataset_id)

@app.get("/api/datasets/{dataset_id}/items")
def dataset_items(dataset_id: str, offset: int = 0, limit: int = 12, finding: str = "", value: str = ""):
    return _dataset_call(datasets.items, dataset_id, max(offset, 0), min(max(limit, 1), 100), finding, value)

@app.get("/api/datasets/{dataset_id}/preview")
def dataset_preview(dataset_id: str, offset: int = 0, limit: int = 12):
    return _dataset_call(datasets.preview, dataset_id, max(offset, 0), min(max(limit, 1), 100))

@app.get("/api/datasets/{dataset_id}/thumbnail/{image_id}")
def dataset_thumbnail(dataset_id: str, image_id: str):
    content = _dataset_call(datasets.thumbnail_bytes, image_id) if dataset_id == datasets.NIH_DATASET_ID else None
    if content is None: raise HTTPException(404, "no thumbnail for this dataset")
    return Response(content=content, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})

@app.get("/api/datasets/{dataset_id}/image/{image_id}")
def dataset_image(dataset_id: str, image_id: str):
    path = _dataset_call(datasets.image_path, dataset_id, image_id)
    return FileResponse(path, media_type="image/png")

@app.post("/api/analyze")
async def analyze(file: UploadFile | None = File(None), mode: str = Form("concise"), provider: str = Form("local"),
                  dataset: str = Form(""), image_id: str = Form("")):
    if mode not in ("concise", "detailed", "evidence"): raise HTTPException(400, "bad mode")
    source = None
    if dataset and image_id:
        path = _dataset_call(datasets.image_path, dataset, image_id)
        try: img = Image.open(path); img.load()
        except Exception: raise HTTPException(400, "unreadable dataset image")
        source = {"dataset": dataset, "dataset_name": datasets.dataset_name(dataset),
                  "image_id": image_id, "findings": datasets.nih_findings(image_id)}
        ground_truth = datasets.nih_ground_truth(image_id)
    elif file is not None:
        if file.content_type not in ("image/png", "image/jpeg"): raise HTTPException(400, "PNG or JPEG only")
        try: img = Image.open(io.BytesIO(await file.read())); img.load()
        except Exception: raise HTTPException(400, "unreadable image")
        ground_truth = None
    else:
        raise HTTPException(400, "provide an image file or dataset + image_id")
    q = quality.screen(img)
    if q["status"] == "flagged" and any("resolution" in i for i in q["issues"]):
        return {"case_id": None, "quality": q, "refused": True, "source": source, "disclaimer": DISCLAIMER}
    res = model.predict(img); res["quality"] = q
    if source:
        res["source"] = source
        res["ground_truth"] = ground_truth
    cid = "CASE-" + uuid.uuid4().hex[:8].upper()
    with db.conn() as c:
        c.execute("insert into cases(id, created, output, model_version, source) values(?,?,?,?,?)",
                  (cid, time.time(), json.dumps({k: v for k, v in res.items() if k != "heatmap_png"}),
                   res["model_version"], f"{dataset}:{image_id}" if source else "upload"))
    res["explanation"] = llm.explain(res, mode, provider)
    return {"case_id": cid, "mode": mode, "disclaimer": DISCLAIMER, "refused": False, **res}

@app.post("/api/explain/{case_id}")  # change mode/provider; predictions are never recomputed
def re_explain(case_id: str, mode: str = Form(...), provider: str = Form(...)):
    with db.conn() as c: r = c.execute("select output from cases where id=?", (case_id,)).fetchone()
    if not r: raise HTTPException(404)
    return llm.explain(json.loads(r["output"]), mode, provider)

@app.post("/api/review")
def review(case_id: str = Form(...), action: str = Form(...), mode: str = Form(""), note: str = Form("")):
    if action not in ("accept","modify","question","reject"): raise HTTPException(400, "bad action")
    with db.conn() as c: c.execute("insert into audit(case_id,action,mode,note,ts) values(?,?,?,?,?)", (case_id, action, mode, note, time.time()))
    return {"ok": True}

@app.get("/api/cases")
def cases():
    with db.conn() as c:
        rows = c.execute("select c.id, c.created, c.model_version, c.source, (select action from audit a where a.case_id=c.id order by a.id desc limit 1) as last_action from cases c order by created desc limit 50").fetchall()
    return [dict(r) for r in rows]

UI_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if UI_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(UI_DIST), html=True), name="ui")
