"""Build demo/data/cases.json from synthetic illustrative cases.

These cases are hand-authored to mimic the competition's input/output structure. They contain
NO competition data (which is not redistributable). Normal-report templates are generic
radiology boilerplate; dictations and reference reports are invented.

Each case is scored with the real local RES scorer (src/res_score.py) so the demo shows the
actual metric, and the pipeline's post-processor (src/postprocess.py) is applied to the raw
"model" text exactly as in production.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from postprocess import reskeleton          # noqa: E402
from res_score import score_case            # noqa: E402

CASES = [
    {
        "id": "cxr-01", "modality": "XRAY", "body_part": "Chest", "study": "XR CHEST 2 VIEWS",
        "template": """FINDINGS:
LUNGS: Lungs are clear. No focal air-space consolidation.
PLEURAL SPACES: No pleural effusion or pneumothorax.
HEART: Cardiac silhouette is within normal size limits.
MEDIASTINUM/HILA: Cardiomediastinal silhouette is within normal limits.
BONES: No acute osseous abnormality identified on this examination.

IMPRESSION:
No acute abnormality.""",
        "dictation": "no effusion or consolidation. mild thoracic spondylosis. cardiomegaly borderline",
        "reference": """FINDINGS:
LUNGS: Lungs are clear. No focal air-space consolidation.
PLEURAL SPACES: No pleural effusion or pneumothorax.
HEART: The cardiac silhouette is borderline enlarged.
MEDIASTINUM/HILA: Cardiomediastinal silhouette is within normal limits.
BONES: Mild thoracic spondylosis is present. No acute osseous abnormality identified on this examination.

IMPRESSION:
1. Borderline cardiomegaly.
2. Mild thoracic spondylosis.
3. No acute cardiopulmonary abnormality.""",
        "model_raw": """FINDINGS:
LUNGS: Lungs are clear. No focal air-space consolidation.
PLEURAL SPACES: No pleural effusion or pneumothorax.
HEART: The cardiac silhouette is borderline enlarged.
MEDIASTINUM/HILA: Cardiomediastinal silhouette is within normal limits.
BONES: Mild thoracic spondylosis is present. No acute osseous abnormality identified on this examination.

IMPRESSION:
1. Borderline cardiomegaly.
2. Mild thoracic spondylosis.
3. No acute cardiopulmonary abnormality.""",
    },
    {
        "id": "hip-01", "modality": "XRAY", "body_part": "Hip", "study": "XR RIGHT HIP 2 VIEWS",
        "template": """FINDINGS:
BONES: No acute fracture or focal osseous lesion.
JOINTS: No dislocation. The joint spaces are normal.
SOFT TISSUES: The soft tissues are unremarkable.

IMPRESSION:
No acute osseous abnormality.""",
        "dictation": "degen changes si joint rt side and rt hip. no fracture",
        "reference": """FINDINGS:
BONES: No acute fracture or focal osseous lesion.

JOINTS: There are degenerative changes of the right hip and right sacroiliac joint. No dislocation.

SOFT TISSUES: The soft tissues are unremarkable.

IMPRESSION:
1. Degenerative changes of the right hip and right sacroiliac joint.
2. No acute osseous abnormality.""",
        "model_raw": """FINDINGS:
BONES: No acute fracture or focal osseous lesion.
JOINTS: There are degenerative changes of the right hip and the right sacroiliac joint. No dislocation. The joint spaces are otherwise normal.
SOFT TISSUES: The soft tissues are unremarkable.

IMPRESSION:
1. Degenerative changes of the right hip and right sacroiliac joint.
2. No acute osseous abnormality.""",
    },
    {
        "id": "hip-02", "modality": "XRAY", "body_part": "Hip", "study": "XR LEFT HIP 2 VIEWS",
        "template": """FINDINGS:
Bones: No acute fracture or focal osseous lesion.
Hip Joints: Normal alignment with preserved joint spaces bilaterally.
Sacroiliac Joints & Pubic Symphysis: Unremarkable.
Soft Tissues: No abnormality identified.

IMPRESSION:
No acute abnormality.""",
        "dictation": "left total hip arthroplasty in situ. exophytic bony changes along left iliac bone and prox left femur",
        "reference": """FINDINGS:
BONES: Exophytic bony changes are noted along the left iliac bone and proximal left femur. No acute fracture or other focal osseous lesion is seen.

HIP JOINTS: Left total hip arthroplasty prostheses are noted in situ. No dislocation is seen.

SACROILIAC JOINTS & PUBIC SYMPHYSIS: Unremarkable.

SOFT TISSUES: No abnormality identified.

IMPRESSION:
1. Left total hip arthroplasty in situ.
2. Exophytic bony changes along the left iliac bone and proximal left femur.""",
        "model_raw": """FINDINGS:
Bones: Exophytic bony changes are noted along the left iliac bone and proximal left femur. No acute fracture or focal osseous lesion.
Hip Joints: Left total hip arthroplasty prostheses are noted in situ. No dislocation.
Sacroiliac Joints & Pubic Symphysis: Unremarkable.
Soft Tissues: No abnormality identified.

IMPRESSION:
1. Left total hip arthroplasty in situ.
2. Exophytic bony changes along the left iliac bone and proximal left femur.""",
    },
    {
        "id": "lspine-01", "modality": "XRAY", "body_part": "Lumbar spine", "study": "XR LUMBAR SPINE 2 VIEWS",
        "template": """FINDINGS:
VERTEBRAE: Normal density and alignment. No fracture or osseous lesion.
DISC SPACES: Preserved.
SOFT TISSUES: Unremarkable.

IMPRESSION:
No acute abnormality evident in the [generic] spine.""",
        "dictation": "mild lower lumbar facet degenerative changes. mild disc space narrowing L4-L5 and L5-S1",
        "reference": """FINDINGS:
VERTEBRAE: Mild lower lumbar facet arthropathy is present. Normal vertebral body heights and alignment. No fracture or osseous lesion.

DISC SPACES: There is mild disc space narrowing at L4-L5 and L5-S1.

SOFT TISSUES: Unremarkable.

IMPRESSION:
1. Mild degenerative changes of the lower lumbar spine, most pronounced at L4-L5 and L5-S1.""",
        "model_raw": """FINDINGS:
VERTEBRAE: Normal density and alignment. No fracture or osseous lesion.
DISC SPACES: There is mild disc space narrowing at L4-L5 and L5-S1.
SOFT TISSUES: Mild lower lumbar facet degenerative changes are present. Unremarkable soft tissues.

IMPRESSION:
1. Mild degenerative changes of the lower lumbar spine.
2. No acute abnormality.""",
    },
    {
        "id": "knee-01", "modality": "XRAY", "body_part": "Knee", "study": "XR RIGHT KNEE 3 VIEWS",
        "template": """FINDINGS:
BONES: No acute fracture or focal osseous lesion.
JOINTS: No dislocation. The joint spaces are normal.
SOFT TISSUES: No joint effusion. The soft tissues are unremarkable.

IMPRESSION:
No acute osseous abnormality.""",
        "dictation": "mod medial compartment jt space narrowing w/ osteophytes. small effusion. no fx",
        "reference": """FINDINGS:
BONES: Marginal osteophytes are present. No acute fracture or focal osseous lesion.

JOINTS: There is moderate medial compartment joint space narrowing. No dislocation.

SOFT TISSUES: There is a small joint effusion. The soft tissues are otherwise unremarkable.

IMPRESSION:
1. Moderate medial compartment osteoarthrosis.
2. Small joint effusion.""",
        "model_raw": """FINDINGS:
BONES: Marginal osteophytes are noted. No acute fracture or focal osseous lesion.
JOINTS: There is moderate medial compartment joint space narrowing. No dislocation.
SOFT TISSUES: There is a small joint effusion. The soft tissues are otherwise unremarkable.

IMPRESSION:
1. Moderate medial compartment degenerative change with joint space narrowing and osteophytes.
2. Small joint effusion.
3. No acute fracture or dislocation.""",
    },
    {
        "id": "ct-abd-01", "modality": "CT", "body_part": "Abdomen", "study": "CT ABDOMEN/PELVIS WITH CONTRAST",
        "template": """FINDINGS:
LIVER: Normal in size and attenuation. No focal hepatic lesion.
GALLBLADDER: Normal. No biliary ductal dilatation.
PANCREAS: Normal in size and contour.
KIDNEYS: Normal appearance without hydronephrosis.
BOWEL: No evidence of bowel obstruction.
PERITONEUM: No ascites. No pneumoperitoneum.

IMPRESSION:
No acute intra-abdominal abnormality identified.""",
        "dictation": "colonic diverticulosis. focal pericolonic fat stranding adjacent to sigmoid, c/w acute diverticulitis. no abscess or free air. gallbladder surgically absent",
        "reference": """FINDINGS:
LIVER: Normal in size and attenuation. No focal hepatic lesion.
GALLBLADDER: Surgically absent. No biliary ductal dilatation.
PANCREAS: Normal in size and contour.
KIDNEYS: Normal appearance without hydronephrosis.
BOWEL: Colonic diverticulosis is present. There is focal pericolonic fat stranding adjacent to the sigmoid colon, consistent with acute diverticulitis. No adjacent abscess or extraluminal free air. No evidence of bowel obstruction.
PERITONEUM: No ascites. No pneumoperitoneum.

IMPRESSION:
1. Acute sigmoid diverticulitis without abscess or perforation.
2. Surgically absent gallbladder.""",
        "model_raw": """FINDINGS:
LIVER: Normal in size and attenuation. No focal hepatic lesion.
GALLBLADDER: The gallbladder is surgically absent. No biliary ductal dilatation.
PANCREAS: Normal in size and contour.
KIDNEYS: Normal appearance without hydronephrosis.
BOWEL: There is colonic diverticulosis with focal pericolonic fat stranding adjacent to the sigmoid colon, consistent with acute diverticulitis. No abscess or free air is seen. No evidence of bowel obstruction.
PERITONEUM: No ascites. No pneumoperitoneum.

IMPRESSION:
1. Acute sigmoid diverticulitis.
2. Surgically absent gallbladder.
3. No bowel obstruction.""",
    },
    {
        "id": "mri-shoulder-01", "modality": "MRI", "body_part": "Shoulder", "study": "MRI RIGHT SHOULDER WITHOUT CONTRAST",
        "template": """FINDINGS:
SUPRASPINATUS: The tendon is intact with normal signal intensity.
INFRASPINATUS: The tendon is intact with normal signal intensity.
SUBSCAPULARIS: The tendon is intact with normal signal intensity.
BICEPS: The tendon is normally located and intact.
GLENOHUMERAL JOINT: No joint effusion. The labrum is intact.
ACROMIOCLAVICULAR JOINT: No significant degenerative change.
BONES: No acute fracture or aggressive osseous lesion.

IMPRESSION:
Normal MRI of the [left/right] shoulder.""",
        "dictation": "mod supraspinatus tendinosis w low-grade bursal-sided partial tear at footprint. small glenohumeral joint effusion. mod AC osteoarthrosis. rest intact",
        "reference": """FINDINGS:
SUPRASPINATUS: There is moderate supraspinatus tendinosis with a low-grade bursal-sided partial-thickness tear at the footprint.
INFRASPINATUS: The infraspinatus tendon is intact.
SUBSCAPULARIS: The subscapularis tendon is intact.
BICEPS: The tendon is normally located and intact.
GLENOHUMERAL JOINT: There is a small glenohumeral joint effusion. The labrum is intact.
ACROMIOCLAVICULAR JOINT: There is moderate acromioclavicular osteoarthrosis.
BONES: No acute fracture or aggressive osseous lesion.

IMPRESSION:
1. Moderate supraspinatus tendinosis with a low-grade bursal-sided partial-thickness tear. No full-thickness tear.
2. Small glenohumeral joint effusion.
3. Moderate acromioclavicular osteoarthrosis.""",
        "model_raw": """FINDINGS:
SUPRASPINATUS: There is moderate supraspinatus tendinosis with low-grade bursal-sided partial-thickness tearing at the footprint. No full-thickness tear.
INFRASPINATUS: The tendon is intact with normal signal intensity.
SUBSCAPULARIS: The tendon is intact with normal signal intensity.
BICEPS: The tendon is normally located and intact.
GLENOHUMERAL JOINT: There is a small joint effusion. The labrum is intact. The articular cartilage is preserved.
ACROMIOCLAVICULAR JOINT: There is moderate acromioclavicular osteoarthrosis with capsular hypertrophy.
BONES: No acute fracture or aggressive osseous lesion. Normal marrow signal.

IMPRESSION:
1. Moderate supraspinatus tendinosis with a low-grade bursal-sided partial-thickness tear.
2. Small glenohumeral joint effusion.
3. Moderate acromioclavicular osteoarthrosis.
4. No full-thickness rotator cuff tear.""",
    },
    {
        "id": "usg-dvt-01", "modality": "USG", "body_part": "Lower extremity veins", "study": "US LOWER EXTREMITY VENOUS DUPLEX, RIGHT",
        "template": """FINDINGS:
DEEP VEINS: The common femoral, femoral, and popliteal veins are compressible with normal color and spectral Doppler flow.
CALF VEINS: The visualized calf veins are patent.
SOFT TISSUES: No popliteal fossa cyst.

IMPRESSION:
No deep venous thrombosis in the [_laterality_] lower extremity.""",
        "dictation": "acute occlusive thrombus in right popliteal and posterior tibial veins. cfv and fv patent",
        "reference": """FINDINGS:
DEEP VEINS: The common femoral and femoral veins are compressible with normal flow. There is acute occlusive thrombus within the right popliteal vein.
CALF VEINS: There is acute occlusive thrombus within the right posterior tibial veins. The remaining visualized calf veins are patent.
SOFT TISSUES: No popliteal fossa cyst.

IMPRESSION:
1. Acute occlusive deep venous thrombosis of the right popliteal and posterior tibial veins.""",
        "model_raw": """FINDINGS:
DEEP VEINS: The common femoral and femoral veins are compressible with normal flow. There is acute occlusive thrombus within the right popliteal vein.
CALF VEINS: There is acute occlusive thrombus within the right posterior tibial veins. The remaining visualized calf veins are patent.
SOFT TISSUES: No popliteal fossa cyst.

IMPRESSION:
1. Acute occlusive deep venous thrombosis of the right popliteal and posterior tibial veins.""",
    },
    {
        "id": "ct-head-01", "modality": "CT", "body_part": "Head", "study": "CT HEAD WITHOUT CONTRAST",
        "template": """FINDINGS:
BRAIN: The brain parenchyma is normal in attenuation. No acute infarction or hemorrhage. No mass effect or midline shift.
VENTRICLES: Normal in size and configuration.
EXTRA-AXIAL SPACES: No extra-axial collection.
BONES: No acute calvarial fracture.

IMPRESSION:
No acute intracranial abnormality.""",
        "dictation": "chronic encephalomalacia right parietal lobe from old infarct. mod periventricular white matter hypodensity - chronic small vessel ischemic change. age-appropriate volume loss. no acute",
        "reference": """FINDINGS:
BRAIN: There is chronic encephalomalacia in the right parietal lobe, consistent with an old infarct. There are moderate periventricular white matter hypodensities, compatible with chronic small vessel ischemic change. Age-appropriate volume loss. No acute infarction or hemorrhage. No mass effect or midline shift.
VENTRICLES: Normal in size and configuration.
EXTRA-AXIAL SPACES: No extra-axial collection.
BONES: No acute calvarial fracture.

IMPRESSION:
1. Chronic right parietal encephalomalacia from prior infarction.
2. Moderate chronic small vessel ischemic change.
3. No acute intracranial abnormality.""",
        "model_raw": """FINDINGS:
BRAIN: There is chronic encephalomalacia in the right parietal lobe from an old infarct. There are moderate periventricular white matter hypodensities, compatible with chronic small vessel ischemic changes. There is age-appropriate volume loss. No acute infarction or hemorrhage. No mass effect or midline shift.
VENTRICLES: Normal in size and configuration.
EXTRA-AXIAL SPACES: No extra-axial collection.
BONES: No acute calvarial fracture.

IMPRESSION:
1. Chronic right parietal encephalomalacia from prior infarction.
2. Moderate chronic small vessel ischemic change.
3. No acute intracranial abnormality.""",
    },
    {
        "id": "foot-01", "modality": "XRAY", "body_part": "Foot", "study": "XR RIGHT FOOT 3 VIEWS",
        "template": """FINDINGS:
BONES: No acute fracture or focal osseous lesion.
JOINTS: No dislocation. The joint spaces are normal.
SOFT TISSUES: The soft tissues are unremarkable.

IMPRESSION:
No acute osseous abnormality.""",
        "dictation": "mild lateral swelling 5th digit at MTP joint. no fracture",
        "reference": """FINDINGS:
BONES: No acute fracture or focal osseous lesion.

JOINTS: No dislocation. The joint spaces are normal.

SOFT TISSUES: There is mild lateral soft tissue swelling of the fifth digit at the metatarsophalangeal joint. The remaining soft tissues are unremarkable.

IMPRESSION:
1. Mild soft tissue swelling of the fifth digit at the metatarsophalangeal joint.
2. No acute osseous abnormality.""",
        "model_raw": """FINDINGS:
BONES: No acute fracture or focal osseous lesion.
JOINTS: There is mild lateral swelling of the fifth digit at the metatarsophalangeal joint. No dislocation. The joint spaces are normal.
SOFT TISSUES: The soft tissues are unremarkable.

IMPRESSION:
1. Mild lateral swelling of the fifth digit at the metatarsophalangeal joint.
2. No acute osseous abnormality.""",
    },
]


def main():
    out = []
    for c in CASES:
        pred = reskeleton(c["template"], c["model_raw"])
        sc = score_case(pred, c["reference"], c["template"])
        b0 = score_case(c["template"], c["reference"], c["template"])
        out.append({
            **{k: c[k] for k in ("id", "modality", "body_part", "study", "template",
                                 "dictation", "reference", "model_raw")},
            "prediction": pred,
            "res": round(sc["res"], 4), "F": round(sc["F"], 4), "I": round(sc["I"], 4),
            "res_baseline": round(b0["res"], 4),
            "fields": [{"label": f[0], "weight": f[1], "score": f[2]} for f in sc["fields"]],
        })
    here = os.path.dirname(__file__)
    json.dump(out, open(os.path.join(here, "data", "cases.json"), "w"), indent=1)
    mean = sum(x["res"] for x in out) / len(out)
    meanb0 = sum(x["res_baseline"] for x in out) / len(out)
    print(f"{len(out)} cases  mean RES {mean:.3f}  (baseline {meanb0:.3f})")
    for x in out:
        print(f"  {x['id']:16s} RES {x['res']:.3f}  (F {x['F']:.2f} / I {x['I']:.2f})  base {x['res_baseline']:.2f}")


if __name__ == "__main__":
    main()
