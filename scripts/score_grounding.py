import json, sys
client = json.load(open(sys.argv[1]))
manifest = {c["file"].split("/")[-1]: c for c in json.load(open(sys.argv[2]))["clips"]}
ok = tot = unfin = 0
for key, g in sorted(client["grounding"].items()):
    clip = manifest[g["clip"]]
    tol = 2 * clip["frame_step_s"]
    a, t = g["answer_s"], g["true_s"]
    good = (a is not None) and abs(a - t) <= tol and not g.get("unfinished", False)
    ok += good; tot += 1
    unfin += g.get("unfinished", False)
print(f"{ok}/{tot} (unfinished: {unfin})")
