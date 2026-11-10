from __future__ import annotations
import csv, json, urllib.request
from collections import defaultdict
from pathlib import Path

START="2026-01-05T00:00Z"
END="2026-01-12T00:00Z"
REGIONS={
  1:("North Scotland","North Scotland Carbon Intensity Forecast"),
  3:("North West England","North West England Carbon Intensity Forecast"),
  7:("South Wales","South Wales Carbon Intensity Forecast"),
  10:("East England","East England Carbon Intensity Forecast"),
  12:("South England","South England Carbon Intensity Forecast"),
  13:("London","London Carbon Intensity Forecast"),
  14:("South East England","South East England Carbon Intensity Forecast"),
}
BASE="https://api.carbonintensity.org.uk/regional/intensity/{start}/{end}/regionid/{rid}"

def extract(payload):
    root=payload.get("data",payload)
    series=root.get("data",[]) if isinstance(root,dict) else root
    out=[]
    for item in series:
        intensity=item.get("intensity",{})
        value=intensity.get("forecast",intensity.get("actual"))
        if value is not None:
            out.append((item["from"],float(value)))
    return out

def main():
    outdir=Path("data/carbon"); outdir.mkdir(parents=True,exist_ok=True)
    by_time=defaultdict(dict); raw={}
    for rid,(name,column) in REGIONS.items():
        url=BASE.format(start=START,end=END,rid=rid)
        req=urllib.request.Request(url,headers={"Accept":"application/json","User-Agent":"QPEO-Reproducibility/1.0"})
        with urllib.request.urlopen(req,timeout=60) as r:
            payload=json.load(r)
        pts=extract(payload)
        if not pts: raise RuntimeError(f"No points for region {rid}")
        raw[str(rid)]={"name":name,"column":column,"url":url,"payload":payload}
        for ts,val in pts: by_time[ts][column]=val
    columns=["Date & Time"]+[v[1] for v in REGIONS.values()]
    rows=[]
    for ts in sorted(by_time):
        row={"Date & Time":ts}; row.update(by_time[ts]); rows.append(row)
    for row in rows:
        missing=[c for c in columns[1:] if c not in row]
        if missing: raise RuntimeError(f"Missing regions at {row['Date & Time']}: {missing}")
    csv_path=outdir/"neso_regional_2026-01-05_2026-01-11.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=columns); w.writeheader(); w.writerows(rows)
    (outdir/"neso_regional_2026-01-05_2026-01-11_raw.json").write_text(json.dumps(raw,indent=2),encoding="utf-8")
    print(f"wrote {len(rows)} synchronized timestamps to {csv_path}")

if __name__=="__main__": main()
