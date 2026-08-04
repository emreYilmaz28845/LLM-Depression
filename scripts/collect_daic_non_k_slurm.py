from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--matrix",type=Path,required=True)
    parser.add_argument("--submission",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    matrix=json.loads(args.matrix.read_text())
    submission=json.loads(args.submission.read_text())
    job_map={}
    for group,payload in submission["groups"].items():
        job_map[str(payload["train_job_id"])]=(group,"train",set(map(int,payload["indices"])))
        job_map[str(payload["evaluation_job_id"])]=(group,"evaluation",set(map(int,payload["indices"])))
    result=subprocess.run(
        ["sacct","-j",",".join(job_map),"--noheader","--parsable2","--format=JobID,JobIDRaw,State,ExitCode,Elapsed,NodeList,AllocCPUS,AllocTRES"],
        text=True,stdout=subprocess.PIPE,check=True,
    )
    tasks={kind:[task for task in matrix["tasks"] if task["kind"]==kind] for kind in ("train","evaluation")}
    rows=[]
    for line in result.stdout.splitlines():
        fields=line.split("|")
        if len(fields)<8: continue
        matched=None
        for job_id in job_map:
            match=re.fullmatch(re.escape(job_id)+r"_(\d+)",fields[0])
            if match: matched=(job_id,int(match.group(1))); break
        if not matched: continue
        job_id,index=matched; group,kind,indices=job_map[job_id]
        if index not in indices or index>=len(tasks[kind]): continue
        task=tasks[kind][index]
        rows.append({"task_id":task["task_id"],"cell_id":task["cell_id"],"group":group,"kind":kind,
                     "job_id":fields[0],"job_id_raw":fields[1],"state":fields[2].split()[0],"exit_code":fields[3],
                     "elapsed":fields[4],"node_list":fields[5],"allocated_cpus":fields[6],"allocated_tres":fields[7]})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in rows))
    print(json.dumps({"output":str(args.output),"rows":len(rows),"expected":len(matrix["tasks"])},sort_keys=True))


if __name__=="__main__": main()
