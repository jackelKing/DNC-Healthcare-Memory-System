"""Synthetic patient data with 5-year timelines and explicit causal links."""
from typing import List, Dict, Any

PATIENTS: List[Dict[str, Any]] = [
    {
        "id": "P001", "name": "Alice Mercer", "age": 54, "sex": "Female",
        "events": [
            {
                "id": "E001", "timestamp": 2019, "condition": "Type 2 Diabetes",
                "symptoms": ["excessive thirst", "frequent urination", "fatigue", "blurred vision"],
                "treatments": ["Metformin 500mg", "dietary counseling", "glucose monitoring"],
                "outcomes": ["HbA1c reduced from 9.2 to 7.8", "weight loss of 4kg"],
                "causal_links": [], "severity": 0.6
            },
            {
                "id": "E002", "timestamp": 2020, "condition": "Hypertension",
                "symptoms": ["headache", "dizziness", "shortness of breath"],
                "treatments": ["Lisinopril 10mg", "sodium restriction", "exercise program"],
                "outcomes": ["BP stabilized at 130/85"],
                "causal_links": [{"cause": "E001", "mechanism": "insulin resistance leads to endothelial dysfunction and elevated BP"}],
                "severity": 0.55
            },
            {
                "id": "E003", "timestamp": 2020, "condition": "Peripheral Neuropathy",
                "symptoms": ["numbness in feet", "tingling sensations", "burning pain at night"],
                "treatments": ["Gabapentin 300mg", "B12 supplementation", "foot care education"],
                "outcomes": ["partial symptom relief", "neuropathy score improved by 30%"],
                "causal_links": [{"cause": "E001", "mechanism": "chronic hyperglycemia damages peripheral nerve myelin sheath"}],
                "severity": 0.5
            },
            {
                "id": "E004", "timestamp": 2021, "condition": "Non-Alcoholic Fatty Liver Disease",
                "symptoms": ["right upper quadrant discomfort", "fatigue", "elevated liver enzymes"],
                "treatments": ["vitamin E supplementation", "caloric restriction", "increased physical activity"],
                "outcomes": ["ALT normalized", "liver steatosis grade reduced on ultrasound"],
                "causal_links": [
                    {"cause": "E001", "mechanism": "insulin resistance promotes hepatic lipid accumulation"},
                    {"cause": "E002", "mechanism": "hypertension exacerbates hepatic inflammation"}
                ],
                "severity": 0.45
            },
            {
                "id": "E005", "timestamp": 2022, "condition": "Diabetic Retinopathy (Mild)",
                "symptoms": ["floaters", "mild visual disturbances", "sensitivity to light"],
                "treatments": ["ophthalmology referral", "stricter glycemic control", "anti-VEGF consideration"],
                "outcomes": ["stable mild NPDR on fundoscopy", "no neovascularization"],
                "causal_links": [
                    {"cause": "E001", "mechanism": "sustained hyperglycemia damages retinal microvasculature"},
                    {"cause": "E002", "mechanism": "hypertension accelerates retinal vessel damage"}
                ],
                "severity": 0.5
            },
            {
                "id": "E006", "timestamp": 2023, "condition": "Chronic Kidney Disease Stage 2",
                "symptoms": ["mild proteinuria", "reduced GFR", "ankle edema"],
                "treatments": ["ACE inhibitor dose increase", "protein restriction", "nephrology referral"],
                "outcomes": ["GFR stabilized at 68 mL/min", "proteinuria reduced"],
                "causal_links": [
                    {"cause": "E001", "mechanism": "diabetic nephropathy from glomerular hyperfiltration"},
                    {"cause": "E002", "mechanism": "hypertension causes glomerulosclerosis and nephron loss"}
                ],
                "severity": 0.6
            },
            {
                "id": "E007", "timestamp": 2023, "condition": "Depression",
                "symptoms": ["persistent sadness", "loss of interest", "sleep disturbance", "poor appetite"],
                "treatments": ["Sertraline 50mg", "cognitive behavioral therapy", "peer support group"],
                "outcomes": ["PHQ-9 score improved from 16 to 8"],
                "causal_links": [
                    {"cause": "E003", "mechanism": "chronic pain from neuropathy contributes to depressive episodes"},
                    {"cause": "E006", "mechanism": "chronic illness burden and disease progression trigger depression"}
                ],
                "severity": 0.55
            }
        ]
    },
    {
        "id": "P002", "name": "Bernard Okafor", "age": 67, "sex": "Male",
        "events": [
            {
                "id": "E101", "timestamp": 2019, "condition": "Coronary Artery Disease",
                "symptoms": ["chest pain on exertion", "dyspnea", "fatigue", "palpitations"],
                "treatments": ["aspirin 81mg", "atorvastatin 40mg", "metoprolol 25mg", "cardiac rehab"],
                "outcomes": ["LVEF 50%", "stress test normalized after rehab"],
                "causal_links": [], "severity": 0.75
            },
            {
                "id": "E102", "timestamp": 2019, "condition": "Hyperlipidemia",
                "symptoms": ["asymptomatic", "elevated LDL on labs"],
                "treatments": ["atorvastatin 40mg", "Mediterranean diet counseling"],
                "outcomes": ["LDL reduced from 188 to 102 mg/dL"],
                "causal_links": [{"cause": "E101", "mechanism": "dyslipidemia is both a cause and consequence of coronary plaque formation"}],
                "severity": 0.4
            },
            {
                "id": "E103", "timestamp": 2020, "condition": "Atrial Fibrillation",
                "symptoms": ["irregular heartbeat", "palpitations", "fatigue", "lightheadedness"],
                "treatments": ["apixaban 5mg BID", "rate control with metoprolol dose increase", "cardiology follow-up"],
                "outcomes": ["rate controlled at less than 80 bpm", "no embolic events"],
                "causal_links": [
                    {"cause": "E101", "mechanism": "ischemic cardiomyopathy promotes atrial remodeling and fibrosis"},
                    {"cause": "E102", "mechanism": "dyslipidemia-related inflammation contributes to atrial structural changes"}
                ],
                "severity": 0.7
            },
            {
                "id": "E104", "timestamp": 2021, "condition": "Heart Failure with Reduced EF",
                "symptoms": ["progressive dyspnea", "orthopnea", "bilateral leg edema", "reduced exercise tolerance"],
                "treatments": ["furosemide 40mg", "sacubitril/valsartan", "spironolactone 25mg", "cardiac resynchronization evaluation"],
                "outcomes": ["LVEF improved to 38%", "NYHA class III to II"],
                "causal_links": [
                    {"cause": "E101", "mechanism": "myocardial ischemia leads to cardiomyocyte loss and ventricular remodeling"},
                    {"cause": "E103", "mechanism": "chronic AF causes tachycardia-mediated cardiomyopathy"}
                ],
                "severity": 0.85
            },
            {
                "id": "E105", "timestamp": 2022, "condition": "Transient Ischemic Attack",
                "symptoms": ["sudden right-sided weakness", "slurred speech lasting 45 min", "confusion"],
                "treatments": ["dual antiplatelet therapy", "carotid ultrasound", "neurology referral"],
                "outcomes": ["full neurological recovery", "no infarct on MRI", "carotid stenosis 40%"],
                "causal_links": [
                    {"cause": "E103", "mechanism": "AF-related thrombus formation in left atrial appendage causes embolic events"},
                    {"cause": "E102", "mechanism": "carotid atherosclerosis from dyslipidemia contributes to cerebrovascular risk"}
                ],
                "severity": 0.8
            },
            {
                "id": "E106", "timestamp": 2023, "condition": "Cardiorenal Syndrome Type 2",
                "symptoms": ["worsening renal function", "reduced urine output", "rising creatinine"],
                "treatments": ["diuretic optimization", "nephrology co-management", "fluid restriction"],
                "outcomes": ["creatinine stabilized at 1.8 mg/dL", "eGFR 42 mL/min"],
                "causal_links": [
                    {"cause": "E104", "mechanism": "reduced cardiac output decreases renal perfusion causing cardiorenal syndrome"},
                    {"cause": "E102", "mechanism": "renal artery atherosclerosis reduces renovascular reserve"}
                ],
                "severity": 0.75
            },
            {
                "id": "E107", "timestamp": 2023, "condition": "Anemia of Chronic Disease",
                "symptoms": ["fatigue", "pallor", "reduced exercise capacity", "Hgb 9.8"],
                "treatments": ["IV iron supplementation", "EPO consideration", "diet optimization"],
                "outcomes": ["Hgb improved to 11.2", "fatigue partially resolved"],
                "causal_links": [
                    {"cause": "E106", "mechanism": "CKD from cardiorenal syndrome reduces erythropoietin production"},
                    {"cause": "E104", "mechanism": "heart failure inflammatory cytokines suppress erythropoiesis"}
                ],
                "severity": 0.5
            },
            {
                "id": "E108", "timestamp": 2024, "condition": "ICD Implantation for Sudden Cardiac Death Prevention",
                "symptoms": ["non-sustained VT on Holter", "LVEF persistently below 35%"],
                "treatments": ["ICD implantation", "electrophysiology follow-up", "amiodarone consideration"],
                "outcomes": ["successful ICD placement", "one appropriate shock in 3 months"],
                "causal_links": [
                    {"cause": "E104", "mechanism": "severely reduced EF with ventricular scar creates substrate for lethal arrhythmias"}
                ],
                "severity": 0.9
            }
        ]
    },
    {
        "id": "P003", "name": "Clara Vasquez", "age": 41, "sex": "Female",
        "events": [
            {
                "id": "E201", "timestamp": 2019, "condition": "Systemic Lupus Erythematosus",
                "symptoms": ["butterfly rash", "joint pain", "fatigue", "photosensitivity", "positive ANA"],
                "treatments": ["hydroxychloroquine 400mg", "sunscreen counseling", "rheumatology referral"],
                "outcomes": ["SLEDAI score reduced from 12 to 6", "rash resolved"],
                "causal_links": [], "severity": 0.7
            },
            {
                "id": "E202", "timestamp": 2019, "condition": "Lupus Nephritis Class III",
                "symptoms": ["hematuria", "proteinuria 2.1g/day", "hypertension", "periorbital edema"],
                "treatments": ["mycophenolate mofetil 2g/day", "prednisone 1mg/kg", "ACE inhibitor"],
                "outcomes": ["proteinuria reduced to 0.4g/day", "renal function preserved eGFR 78"],
                "causal_links": [
                    {"cause": "E201", "mechanism": "immune complex deposition in glomeruli causes inflammatory nephritis"}
                ],
                "severity": 0.75
            },
            {
                "id": "E203", "timestamp": 2020, "condition": "Avascular Necrosis of Left Hip",
                "symptoms": ["progressive left hip pain", "limping", "reduced ROM", "pain at rest"],
                "treatments": ["core decompression surgery", "bisphosphonate therapy", "physical therapy"],
                "outcomes": ["pain reduced by 60%", "avoided total hip replacement"],
                "causal_links": [
                    {"cause": "E202", "mechanism": "prolonged high-dose corticosteroids impair osteoblast function causing AVN"}
                ],
                "severity": 0.65
            },
            {
                "id": "E204", "timestamp": 2021, "condition": "Antiphospholipid Syndrome",
                "symptoms": ["DVT in left leg", "livedo reticularis", "recurrent pregnancy loss"],
                "treatments": ["warfarin with target INR 2.5-3.5", "hematology referral", "thrombophilia workup"],
                "outcomes": ["no recurrent thrombosis on anticoagulation", "INR therapeutic"],
                "causal_links": [
                    {"cause": "E201", "mechanism": "SLE-associated autoantibodies cause hypercoagulability"}
                ],
                "severity": 0.8
            },
            {
                "id": "E205", "timestamp": 2022, "condition": "Opportunistic Infection - Pneumocystis Pneumonia",
                "symptoms": ["progressive dyspnea", "dry cough", "fever", "hypoxia SpO2 88%"],
                "treatments": ["TMP-SMX high dose 21 days", "prednisone taper", "ICU admission"],
                "outcomes": ["full recovery in 3 weeks", "no residual lung damage"],
                "causal_links": [
                    {"cause": "E202", "mechanism": "mycophenolate and corticosteroids cause profound immunosuppression enabling PCP"}
                ],
                "severity": 0.9
            },
            {
                "id": "E206", "timestamp": 2022, "condition": "Steroid-Induced Diabetes Mellitus",
                "symptoms": ["hyperglycemia on labs", "polydipsia", "polyuria", "recurrent infections"],
                "treatments": ["insulin therapy", "prednisone dose reduction", "endocrinology referral"],
                "outcomes": ["HbA1c 7.2", "insulin requirement decreasing with steroid taper"],
                "causal_links": [
                    {"cause": "E202", "mechanism": "chronic corticosteroid use increases hepatic gluconeogenesis and induces insulin resistance"}
                ],
                "severity": 0.55
            },
            {
                "id": "E207", "timestamp": 2023, "condition": "Cerebral Venous Sinus Thrombosis",
                "symptoms": ["severe headache", "visual changes", "papilledema", "seizure"],
                "treatments": ["anticoagulation intensification", "neurology ICU", "acetazolamide for ICP"],
                "outcomes": ["full neurological recovery", "sinus recanalization on MRV at 6 months"],
                "causal_links": [
                    {"cause": "E204", "mechanism": "antiphospholipid antibodies promote cerebral venous thrombosis despite anticoagulation"},
                    {"cause": "E201", "mechanism": "active SLE flare with elevated inflammatory markers raises thrombotic risk"}
                ],
                "severity": 0.95
            }
        ]
    }
]

def get_patient_by_id(patient_id: str) -> Dict[str, Any]:
    for p in PATIENTS:
        if p["id"] == patient_id:
            return p
    raise KeyError(f"Patient {patient_id} not found")

def get_all_patient_ids() -> List[str]:
    return [p["id"] for p in PATIENTS]

def get_patient_names() -> Dict[str, str]:
    return {p["id"]: p["name"] for p in PATIENTS}
