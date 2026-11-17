import json, os, re, urllib.request
KB = [
 ("NIH ChestX-ray14 documentation", "Labels in ChestX-ray14 were text-mined from radiology reports using NLP. They are research labels, not individually confirmed diagnoses."),
 ("Grad-CAM (Selvaraju et al., 2017)", "Grad-CAM highlights image regions that influenced a model output. It shows model attention, not proof that disease is present."),
 ("Calibration (Guo et al., 2017)", "Raw sigmoid scores from neural networks are often over-confident; temperature scaling on a validation set improves calibration."),
 ("Effusion label note", "The dataset label Effusion is an effusion-related research label, not a confirmed clinical diagnosis of pleural effusion."),
 ("Pneumothorax / Pneumonia label note", "Pneumonia and Pneumothorax labels in public datasets can be noisy; radiologist review is required for any clinical interpretation."),
]
def retrieve(query, k=3):
    w = set(re.findall(r"\w+", query.lower()))
    s = sorted(KB, key=lambda d: -len(w & set(re.findall(r"\w+", (d[0]+" "+d[1]).lower()))))
    return [{"source": a, "text": b} for a,b in s[:k]]

def _prompt(result, mode, refs):
    f = "; ".join(f"{x['label']} {x['score']:.2f}{' (uncertain)' if x['uncertain'] else ''}" for x in result["findings"])
    style = {"concise": "Write 3 short bullet points.", "detailed": "Write a detailed explanation in 3 short paragraphs, including limitations.",
             "evidence": "Write structured points, each ending with a citation like [1]."}[mode]
    r = "\n".join(f"[{i+1}] {d['source']}: {d['text']}" for i,d in enumerate(refs))
    return (f"You assist clinicians in a RESEARCH prototype. Model scores: {f}. Quality: {result['quality']['status']}. Untrained demo model: {not result['trained']}.\n"
            f"Use ONLY these references and never invent citations, diagnoses or treatment advice:\n{r}\n{style}")

def _ollama(p):
    body = json.dumps({"model": os.getenv("OLLAMA_MODEL","qwen2.5:1.5b"), "prompt": p, "stream": False}).encode()
    url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/generate"
    req = urllib.request.Request(url, body, {"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req, timeout=180))["response"]

def _external(p):  # any OpenAI-compatible endpoint
    base = os.getenv("LLM_API_BASE","https://api.openai.com/v1"); key = os.environ["LLM_API_KEY"]
    body = json.dumps({"model": os.getenv("LLM_API_MODEL","gpt-4o-mini"), "messages":[{"role":"user","content":p}]}).encode()
    req = urllib.request.Request(base+"/chat/completions", body, {"Content-Type":"application/json","Authorization":"Bearer "+key})
    return json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]["content"]

def _template(result, refs):
    lines = [f"- {x['label']}: score {x['score']:.2f}" + (" (uncertain, needs review)" if x["uncertain"] else "") for x in result["findings"]]
    if not result["trained"]: lines.append("- WARNING: model is untrained; scores are not meaningful.")
    return "\n".join(lines)

def explain(result, mode, provider):
    refs = retrieve(" ".join(x["label"] for x in result["findings"]) + " heatmap calibration label")
    used, note = provider, ""
    try:
        text = _ollama(_prompt(result, mode, refs)) if provider == "local" else _external(_prompt(result, mode, refs))
    except Exception as e:
        text, used, note = _template(result, refs), "template-fallback", f"{provider} LLM unavailable ({type(e).__name__})"
    return {"text": text, "provider": used, "note": note, "references": refs if mode != "concise" else refs[:1]}
