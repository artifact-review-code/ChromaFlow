#!/usr/bin/env python3
"""Extract reviewer-facing CSV tables from the final LaTeX manuscript."""
from __future__ import annotations
import argparse, csv, io, re
from pathlib import Path

DATASETS=("MoNuSeg","CPM-17","CoNIC","NuInsSeg"); IN_METRICS=("AJI","Dice","DQ","SQ","PQ","Recall")
DIRECTIONS=("MO->CPM","CPM->MO","Co->Nu","Nu->Co","MO->Co","MO->Nu","CPM->Co","CPM->Nu")
CROSS_METRICS=("AJI","Dice","DQ","PQ")

def method(cell):
    cell=re.sub(r"\\cite\{[^}]+\}","",cell)
    cell=re.sub(r"\\(?:textbf|underline)\{([^{}]+)\}",r"\1",cell).replace("~"," ").strip()
    return "ChromaFlow" if cell in ("Ours", "ChromaFlow (Ours)") else cell
def value(cell):
    if "--" in cell:return ""
    m=re.findall(r"(?<![A-Za-z])\d+\.\d+",cell); return m[-1] if m else ""
def section(text,start,end):
    a=text.index(start); return text[a:text.index(end,a)].splitlines()
def rows(lines,n):
    group=""; pending=[]
    for line in lines:
        if ("Fully supervised references" in line or "Fully supervised methods" in line):group="fully_supervised"
        elif ("Pretrained foundation-model references" in line or "Pretrained foundation-model-based methods" in line):group="foundation_model"
        elif "Weakly supervised methods" in line:group="weakly_supervised"
        if not group:continue
        if pending or "&" in line:pending.append(line.strip())
        if not pending or not line.rstrip().endswith(r"\\"):continue
        joined=" ".join(pending); pending=[]
        cells=[x.strip() for x in joined.rsplit(r"\\",1)[0].split("&")]
        if len(cells)==n+1:yield group,method(cells[0]),[value(x) for x in cells[1:]]
def output(fields,data):
    s=io.StringIO(newline=""); w=csv.DictWriter(s,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(data); return s.getvalue()
def extract(path):
    text=re.sub(r"(?<!\\)%[^\n]*", "", path.read_text()); inside=[]; cross=[]
    for group,name,vals in rows(section(text,r"\newcommand{\indomaintable}",r"\newcommand{\crossdomaintable}"),26):
        params,gflops,*metrics=vals
        for di,dataset in enumerate(DATASETS):
            for mi,metric in enumerate(IN_METRICS):inside.append(dict(group=group,method=name,params_m=params,gflops=gflops,dataset=dataset,metric=metric,value=metrics[di*6+mi]))
    for group,name,vals in rows(section(text,r"\label{tab:cross}",r"\end{table*}"),32):
        for di,direction in enumerate(DIRECTIONS):
            for mi,metric in enumerate(CROSS_METRICS):cross.append(dict(group=group,method=name,direction=direction,metric=metric,value=vals[di*4+mi]))
    if not inside or not cross:raise RuntimeError("Could not find both result tables")
    return output(["group","method","params_m","gflops","dataset","metric","value"],inside),output(["group","method","direction","metric","value"],cross)
def main():
    p=argparse.ArgumentParser(); p.add_argument("manuscript",type=Path); p.add_argument("--output-dir",type=Path,default=Path("results")); p.add_argument("--check",action="store_true"); a=p.parse_args(); one,two=extract(a.manuscript)
    targets={a.output_dir/"table1_indomain.csv":one,a.output_dir/"table2_crossdomain.csv":two}
    if a.check:
        bad=[str(k) for k,v in targets.items() if not k.exists() or k.read_text()!=v]
        if bad:raise SystemExit("Result table drift detected: "+", ".join(bad))
        print("Result CSV files match the LaTeX manuscript."); return
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for k,v in targets.items():k.write_text(v); print("wrote",k)
if __name__=="__main__":main()
