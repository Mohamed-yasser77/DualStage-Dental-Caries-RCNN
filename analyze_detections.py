import json
import numpy as np

with open("test_10_results/summary.json") as f:
    data = json.load(f)

print(f"{'Image':<12} | {'Teeth':<6} | {'Raw Lesions':<12} | {'>=50px':<8} | {'>=200px':<8} | {'Max Lesion px':<14} | {'Total Caries px':<16}")
print("-" * 85)
for x in data:
    n_raw = x['num_lesions_detected']
    sig50 = sum(1 for l in x['lesions'] if l['area_px'] >= 50)
    sig200 = sum(1 for l in x['lesions'] if l['area_px'] >= 200)
    max_px = max([l['area_px'] for l in x['lesions']]) if x['lesions'] else 0
    tot_px = x['caries_pixel_area']
    print(f"{x['image']:<12} | {x['num_teeth_detected']:<6} | {n_raw:<12} | {sig50:<8} | {sig200:<8} | {max_px:<14} | {tot_px:<16}")
