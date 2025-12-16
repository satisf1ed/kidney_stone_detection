import json

p = "data/coco/annotations_val.json"
d = json.load(open(p))
cats = sorted({c["id"] for c in d["categories"]})
ann = sorted({a["category_id"] for a in d["annotations"]})[:20]

print("categories ids:", cats[:20], "...", "min=",min(cats), "max=",max(cats))
print("annotation category_id sample:", ann, "...", "min=",min({a["category_id"] for a in d["annotations"]}))
