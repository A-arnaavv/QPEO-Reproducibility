from __future__ import annotations
import csv, json, urllib.request
from pathlib import Path

DATE_FROM = "2025-01-15T00:00Z"
DATE_TO = "2025-01-16T00:00Z"
REGIONS = {1:"North Scotland",3:"North West England",10:"East England",13:"London"}
BASE = "https://api.carbonintensity.org.uk/regional/intensity/{from_}/{to_}/regionid/{region}"

def extract_points(payload):
    root = payload.get("data", payload)
    if isinstance(root, dict):
        series = root.get("data", root.get("values", []))
    else:
        series = root
    out=[]
    for item in series:
        intensity=item.get("intensity",{}) if isinstance(item,dict) else {}
        value=intensity.get("forecast", intensity.get("actual"))
        if value is None: continue
        out.append((item.get("from"),item.get("to"),float(value)))
    return out

def main():
    outdir=Path("data/carbon"); outdir.mkdir(parents=True,exist_ok=True)
    rows=[]; raw={}
    for rid,name in REGIONS.items():
        url=BASE.format(from_=DATE_FROM,to_=DATE_TO,region=rid)
        req=urllib.request.Request(url,headers={"Accept":"application/json","User-Agent":"QPEO-Reproducibility/1.0"})
        with urllib.request.urlopen(req,timeout=30) as r:
            payload=json.load(r)
        pts=extract_points(payload)
        if not pts: raise RuntimeError(f"No carbon-intensity points returned for region {rid}")
        raw[str(rid)]={"name":name,"url":url,"payload":payload}
        for start,end,val in pts:
            rows.append({"region_id":rid,"region_name":name,"from":start,"to":end,"forecast_gco2_per_kwh":val})
    (outdir/"neso_regional_2025-01-15_raw.json").write_text(json.dumps(raw,indent=2),encoding="utf-8")
    with (outdir/"neso_regional_2025-01-15.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["region_id","region_name","from","to","forecast_gco2_per_kwh"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} points")

if __name__=="__main__": main()
