# ================================================================
# MDS1 — Multimodal Medical Report Analyzer
# Robust rule-based laboratory report parser
# Languages: English / Hindi / Marathi
# Educational prototype — NOT a medical diagnosis system.
# ================================================================

import os
import re
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pandas as pd
except ImportError:
    pd = None


# ----------------------------------------------------------------
# OCR setup
# ----------------------------------------------------------------
if pytesseract:
    if os.name == "nt":
        for tpath in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ):
            if os.path.exists(tpath):
                pytesseract.pytesseract.tesseract_cmd = tpath
                break
    elif os.path.exists("/usr/bin/tesseract"):
        pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


# ----------------------------------------------------------------
# Reference ranges
# NOTE: These are simplified prototype ranges. Real ranges vary by
# laboratory, age, sex, pregnancy status and clinical context.
# ----------------------------------------------------------------
REFERENCE_RANGES = {
    "hemoglobin": {"min": 12.0, "max": 17.5, "unit": "g/dL"},
    "wbc": {"min": 4000, "max": 11000, "unit": "/uL"},
    "rbc": {"min": 4.0, "max": 6.0, "unit": "million/uL"},
    "platelets": {"min": 150000, "max": 450000, "unit": "/uL"},
    "hematocrit": {"min": 36.0, "max": 52.0, "unit": "%"},
    "mcv": {"min": 80.0, "max": 100.0, "unit": "fL"},
    "mch": {"min": 27.0, "max": 33.0, "unit": "pg"},
    "mchc": {"min": 32.0, "max": 36.0, "unit": "g/dL"},
    "fasting_glucose": {"min": 70, "max": 99, "unit": "mg/dL"},
    "random_glucose": {"min": 70, "max": 199, "unit": "mg/dL"},
    "postprandial_glucose": {"min": 70, "max": 139, "unit": "mg/dL"},
    "hba1c": {"min": 4.0, "max": 5.6, "unit": "%"},
    "alt": {"min": 7, "max": 56, "unit": "U/L"},
    "ast": {"min": 10, "max": 40, "unit": "U/L"},
    "alp": {"min": 44, "max": 147, "unit": "U/L"},
    "bilirubin_total": {"min": 0.1, "max": 1.2, "unit": "mg/dL"},
    "bilirubin_direct": {"min": 0.0, "max": 0.3, "unit": "mg/dL"},
    "albumin": {"min": 3.5, "max": 5.0, "unit": "g/dL"},
    "total_protein": {"min": 6.0, "max": 8.3, "unit": "g/dL"},
    "creatinine": {"min": 0.6, "max": 1.3, "unit": "mg/dL"},
    "urea": {"min": 15, "max": 45, "unit": "mg/dL"},
    "bun": {"min": 7, "max": 20, "unit": "mg/dL"},
    "uric_acid": {"min": 3.0, "max": 7.0, "unit": "mg/dL"},
    "total_cholesterol": {"min": 0, "max": 199, "unit": "mg/dL"},
    "ldl": {"min": 0, "max": 99, "unit": "mg/dL"},
    "hdl": {"min": 40, "max": 100, "unit": "mg/dL"},
    "triglycerides": {"min": 0, "max": 149, "unit": "mg/dL"},
    "tsh": {"min": 0.4, "max": 4.0, "unit": "mIU/L"},
    "free_t4": {"min": 0.8, "max": 1.8, "unit": "ng/dL"},
    "free_t3": {"min": 2.3, "max": 4.2, "unit": "pg/mL"},
    "sodium": {"min": 135, "max": 145, "unit": "mmol/L"},
    "potassium": {"min": 3.5, "max": 5.0, "unit": "mmol/L"},
    "calcium": {"min": 8.5, "max": 10.5, "unit": "mg/dL"},
    "ferritin": {"min": 15, "max": 150, "unit": "ng/mL"},
    "serum_iron": {"min": 60, "max": 170, "unit": "ug/dL"},
}


# ----------------------------------------------------------------
# Aliases
# ----------------------------------------------------------------
TEST_ALIASES = {
    "hemoglobin": ["hemoglobin", "haemoglobin", "hemoglobin hb", "haemoglobin hb", "hb"],
    "wbc": [
        "wbc",
        "white blood cell count",
        "white blood cell",
        "white blood cells",
        "total leukocyte count",
        "total leucocyte count",
        "tlc",
    ],
    "rbc": ["rbc", "red blood cell count", "red blood cell", "red blood cells"],
    "platelets": ["platelet count", "platelets", "platelet", "plt"],
    "hematocrit": ["hematocrit", "haematocrit", "hct", "pcv", "packed cell volume"],
    "mcv": ["mcv", "mean corpuscular volume"],
    "mch": ["mch", "mean corpuscular hemoglobin", "mean corpuscular haemoglobin"],
    "mchc": [
        "mchc",
        "mean corpuscular hemoglobin concentration",
        "mean corpuscular haemoglobin concentration",
    ],
    "fasting_glucose": [
        "fasting glucose",
        "fasting blood glucose",
        "fasting plasma glucose",
        "fbs",
        "fbg",
    ],
    "random_glucose": [
        "random glucose",
        "random blood glucose",
        "random plasma glucose",
        "rbs",
    ],
    "postprandial_glucose": [
        "postprandial glucose",
        "post meal glucose",
        "post-meal glucose",
        "ppbs",
        "pp glucose",
    ],
    "hba1c": [
        "hba1c",
        "hb a1c",
        "glycated hemoglobin",
        "glycated haemoglobin",
        "glycosylated hemoglobin",
    ],
    "alt": ["alt", "sgpt", "alanine aminotransferase"],
    "ast": ["ast", "sgot", "aspartate aminotransferase"],
    "alp": ["alp", "alkaline phosphatase"],
    "bilirubin_total": ["total bilirubin", "bilirubin total", "serum total bilirubin"],
    "bilirubin_direct": ["direct bilirubin", "bilirubin direct", "serum direct bilirubin"],
    "albumin": ["albumin", "serum albumin"],
    "total_protein": ["total protein", "serum total protein"],
    "creatinine": ["creatinine", "serum creatinine"],
    "urea": ["blood urea", "serum urea", "urea"],
    "bun": ["bun", "blood urea nitrogen"],
    "uric_acid": ["uric acid", "serum uric acid"],
    "total_cholesterol": ["total cholesterol", "cholesterol total", "serum cholesterol"],
    "ldl": ["ldl cholesterol", "ldl"],
    "hdl": ["hdl cholesterol", "hdl"],
    "triglycerides": ["triglycerides", "triglyceride"],
    "tsh": ["tsh", "thyroid stimulating hormone"],
    "free_t4": ["free t4", "ft4", "free thyroxine"],
    "free_t3": ["free t3", "ft3", "free triiodothyronine"],
    "sodium": ["serum sodium", "sodium"],
    "potassium": ["serum potassium", "potassium"],
    "calcium": ["serum calcium", "calcium"],
    "ferritin": ["serum ferritin", "ferritin"],
    "serum_iron": ["serum iron", "iron"],
}


# ----------------------------------------------------------------
# Human-readable names
# ----------------------------------------------------------------
DISPLAY_NAMES = {
    "hemoglobin": "Hemoglobin (Hb)",
    "wbc": "White Blood Cells (WBC)",
    "rbc": "Red Blood Cells (RBC)",
    "platelets": "Platelet Count",
    "hematocrit": "Hematocrit (HCT)",
    "mcv": "Mean Corpuscular Volume (MCV)",
    "mch": "Mean Corpuscular Hemoglobin (MCH)",
    "mchc": "Mean Corpuscular Hemoglobin Concentration (MCHC)",
    "fasting_glucose": "Fasting Glucose",
    "random_glucose": "Random Glucose",
    "postprandial_glucose": "Postprandial Glucose",
    "hba1c": "HbA1c",
    "alt": "ALT (SGPT)",
    "ast": "AST (SGOT)",
    "alp": "Alkaline Phosphatase (ALP)",
    "bilirubin_total": "Total Bilirubin",
    "bilirubin_direct": "Direct Bilirubin",
    "albumin": "Albumin",
    "total_protein": "Total Protein",
    "creatinine": "Serum Creatinine",
    "urea": "Blood Urea",
    "bun": "Blood Urea Nitrogen (BUN)",
    "uric_acid": "Uric Acid",
    "total_cholesterol": "Total Cholesterol",
    "ldl": "LDL Cholesterol",
    "hdl": "HDL Cholesterol",
    "triglycerides": "Triglycerides",
    "tsh": "TSH",
    "free_t4": "Free T4",
    "free_t3": "Free T3",
    "sodium": "Sodium",
    "potassium": "Potassium",
    "calcium": "Calcium",
    "ferritin": "Ferritin",
    "serum_iron": "Serum Iron",
}


# ----------------------------------------------------------------
# Preserved richer explanations for the original core tests
# ----------------------------------------------------------------
PREDEFINED_EXPLANATIONS = {
    "hemoglobin": {
        "en": {
            "name": "Hemoglobin (Hb)",
            "LOW": {
                "medical": "Hemoglobin is below the configured reference range. This pattern may be evaluated in the context of iron deficiency, anemia or blood loss.",
                "simple": "💡 Hemoglobin carries oxygen in your blood. A lower result can be associated with fatigue or weakness.",
                "conclusion": "📌 Discuss the result with your healthcare professional and consider the full blood profile.",
            },
            "HIGH": {
                "medical": "Hemoglobin is above the configured reference range. Elevated values can occur with dehydration and other clinical factors.",
                "simple": "💡 The blood concentration of hemoglobin is higher than the configured range.",
                "conclusion": "📌 Review hydration and clinical context with your healthcare professional.",
            },
            "NORMAL": {
                "medical": "Hemoglobin is within the configured reference range.",
                "simple": "💡 The measured oxygen-carrying protein is within the configured range.",
                "conclusion": "📌 Continue routine health monitoring as appropriate.",
            },
        },
        "hi": {
            "name": "हीमोग्लोबिन (Hb)",
            "LOW": {
                "medical": "हीमोग्लोबिन निर्धारित संदर्भ सीमा से कम है। इसका मूल्यांकन एनीमिया, आयरन की कमी या अन्य कारणों के संदर्भ में किया जाता है।",
                "simple": "💡 हीमोग्लोबिन रक्त में ऑक्सीजन पहुंचाता है। कम स्तर के साथ थकान या कमजोरी हो सकती है।",
                "conclusion": "📌 इस परिणाम पर चिकित्सक से चर्चा करें।",
            },
            "HIGH": {
                "medical": "हीमोग्लोबिन निर्धारित संदर्भ सीमा से अधिक है।",
                "simple": "💡 रक्त में हीमोग्लोबिन का स्तर निर्धारित सीमा से अधिक है।",
                "conclusion": "📌 परिणाम को चिकित्सकीय संदर्भ में समझने के लिए डॉक्टर से चर्चा करें।",
            },
            "NORMAL": {
                "medical": "हीमोग्लोबिन निर्धारित संदर्भ सीमा के भीतर है।",
                "simple": "💡 हीमोग्लोबिन का स्तर निर्धारित सीमा में है।",
                "conclusion": "📌 नियमित स्वास्थ्य निगरानी जारी रखें।",
            },
        },
        "mr": {
            "name": "हिमोग्लोबिन (Hb)",
            "LOW": {
                "medical": "हिमोग्लोबिन निर्धारित संदर्भ मर्यादेपेक्षा कमी आहे.",
                "simple": "💡 हिमोग्लोबिन रक्तामध्ये ऑक्सिजन वाहून नेते. कमी पातळीमुळे थकवा किंवा अशक्तपणा जाणवू शकतो.",
                "conclusion": "📌 या निकालाबाबत डॉक्टरांचा सल्ला घ्या.",
            },
            "HIGH": {
                "medical": "हिमोग्लोबिन निर्धारित संदर्भ मर्यादेपेक्षा जास्त आहे.",
                "simple": "💡 रक्तातील हिमोग्लोबिनची पातळी निर्धारित मर्यादेपेक्षा जास्त आहे.",
                "conclusion": "📌 निकालाचा वैद्यकीय संदर्भ डॉक्टरांसोबत तपासा.",
            },
            "NORMAL": {
                "medical": "हिमोग्लोबिन निर्धारित संदर्भ मर्यादेत आहे.",
                "simple": "💡 हिमोग्लोबिनची पातळी निर्धारित मर्यादेत आहे.",
                "conclusion": "📌 नियमित आरोग्य तपासणी सुरू ठेवा.",
            },
        },
    },
    "wbc": {
        "en": {
            "name": "White Blood Cells (WBC)",
            "LOW": {
                "medical": "WBC is below the configured reference range.",
                "simple": "💡 White blood cells help the body respond to infections.",
                "conclusion": "📌 Discuss the result with a healthcare professional, especially if symptoms are present.",
            },
            "HIGH": {
                "medical": "WBC is above the configured reference range and can occur with infection, inflammation or other factors.",
                "simple": "💡 The immune system may be responding to an inflammatory or infectious process.",
                "conclusion": "📌 Review the result together with symptoms and clinical history.",
            },
            "NORMAL": {
                "medical": "WBC is within the configured reference range.",
                "simple": "💡 The measured white blood-cell count is within the configured range.",
                "conclusion": "📌 Continue routine monitoring as appropriate.",
            },
        },
        "hi": {
            "name": "श्वेत रक्त कोशिकाएं (WBC)",
            "LOW": {
                "medical": "WBC निर्धारित संदर्भ सीमा से कम है।",
                "simple": "💡 WBC संक्रमण से लड़ने में मदद करते हैं।",
                "conclusion": "📌 इस परिणाम पर चिकित्सक से चर्चा करें।",
            },
            "HIGH": {
                "medical": "WBC निर्धारित संदर्भ सीमा से अधिक है और संक्रमण या सूजन जैसे कई कारणों से हो सकता है।",
                "simple": "💡 शरीर में प्रतिरक्षा प्रतिक्रिया सक्रिय हो सकती है।",
                "conclusion": "📌 लक्षणों और मेडिकल इतिहास के साथ डॉक्टर से परिणाम की समीक्षा करें।",
            },
            "NORMAL": {
                "medical": "WBC निर्धारित संदर्भ सीमा के भीतर है।",
                "simple": "💡 श्वेत रक्त कोशिकाओं की संख्या निर्धारित सीमा में है।",
                "conclusion": "📌 नियमित स्वास्थ्य निगरानी जारी रखें।",
            },
        },
        "mr": {
            "name": "पांढऱ्या रक्तपेशी (WBC)",
            "LOW": {
                "medical": "WBC निर्धारित संदर्भ मर्यादेपेक्षा कमी आहे.",
                "simple": "💡 WBC शरीराला संसर्गाशी लढण्यास मदत करतात.",
                "conclusion": "📌 या निकालाबाबत डॉक्टरांचा सल्ला घ्या.",
            },
            "HIGH": {
                "medical": "WBC निर्धारित संदर्भ मर्यादेपेक्षा जास्त आहे.",
                "simple": "💡 शरीरामध्ये संसर्ग किंवा दाहाविरुद्ध प्रतिकारशक्तीची प्रतिक्रिया असू शकते.",
                "conclusion": "📌 लक्षणे व वैद्यकीय इतिहासासह डॉक्टरांकडून तपासणी करून घ्या.",
            },
            "NORMAL": {
                "medical": "WBC निर्धारित संदर्भ मर्यादेत आहे.",
                "simple": "💡 पांढऱ्या रक्तपेशींची संख्या निर्धारित मर्यादेत आहे.",
                "conclusion": "📌 नियमित आरोग्य तपासणी सुरू ठेवा.",
            },
        },
    },
    "creatinine": {
        "en": {
            "name": "Serum Creatinine",
            "LOW": {
                "medical": "Creatinine is below the configured reference range.",
                "simple": "💡 Creatinine is a waste product related to muscle metabolism; a low result can have several benign explanations.",
                "conclusion": "📌 Interpret it together with kidney function and clinical context.",
            },
            "HIGH": {
                "medical": "Creatinine is above the configured reference range and may be associated with reduced renal filtration or other factors.",
                "simple": "💡 The kidneys help remove creatinine from the blood.",
                "conclusion": "📌 Discuss the result with a healthcare professional, particularly when persistent or accompanied by symptoms.",
            },
            "NORMAL": {
                "medical": "Serum creatinine is within the configured reference range.",
                "simple": "💡 The measured creatinine level is within this prototype's reference interval.",
                "conclusion": "📌 Interpret kidney health using the complete clinical picture.",
            },
        },
        "hi": {
            "name": "सीरम क्रिएटिनाइन",
            "LOW": {
                "medical": "क्रिएटिनाइन निर्धारित संदर्भ सीमा से कम है।",
                "simple": "💡 क्रिएटिनाइन मांसपेशियों के मेटाबॉलिज्म से संबंधित अपशिष्ट पदार्थ है।",
                "conclusion": "📌 पूरे किडनी प्रोफाइल के साथ डॉक्टर से चर्चा करें।",
            },
            "HIGH": {
                "medical": "क्रिएटिनाइन निर्धारित संदर्भ सीमा से अधिक है और किडनी फिल्ट्रेशन से संबंधित हो सकता है।",
                "simple": "💡 किडनी रक्त से क्रिएटिनाइन को बाहर निकालने में मदद करती है।",
                "conclusion": "📌 इस परिणाम पर चिकित्सक से चर्चा करें।",
            },
            "NORMAL": {
                "medical": "सीरम क्रिएटिनाइन निर्धारित संदर्भ सीमा के भीतर है।",
                "simple": "💡 क्रिएटिनाइन का मान निर्धारित सीमा में है।",
                "conclusion": "📌 किडनी स्वास्थ्य को पूरे क्लिनिकल संदर्भ के साथ समझें।",
            },
        },
        "mr": {
            "name": "सिरम क्रिएटिनाइन",
            "LOW": {
                "medical": "क्रिएटिनाइन निर्धारित संदर्भ मर्यादेपेक्षा कमी आहे.",
                "simple": "💡 क्रिएटिनाइन स्नायूंच्या मेटाबॉलिझमशी संबंधित अपशिष्ट पदार्थ आहे.",
                "conclusion": "📌 संपूर्ण किडनी प्रोफाइलसोबत डॉक्टरांचा सल्ला घ्या.",
            },
            "HIGH": {
                "medical": "क्रिएटिनाइन निर्धारित संदर्भ मर्यादेपेक्षा जास्त आहे.",
                "simple": "💡 किडनी रक्तातील क्रिएटिनाइन बाहेर काढण्यास मदत करते.",
                "conclusion": "📌 या निकालाबाबत डॉक्टरांचा सल्ला घ्या.",
            },
            "NORMAL": {
                "medical": "सिरम क्रिएटिनाइन निर्धारित संदर्भ मर्यादेत आहे.",
                "simple": "💡 क्रिएटिनाइनची पातळी निर्धारित मर्यादेत आहे.",
                "conclusion": "📌 किडनीचे आरोग्य संपूर्ण वैद्यकीय संदर्भासह समजून घ्या.",
            },
        },
    },
}


# ----------------------------------------------------------------
# Robust numeric parsing
# ----------------------------------------------------------------
NUMBER_RE = re.compile(
    r"""
    (?<![A-Za-z])
    [-+]?
    (?:
        \d{1,3}(?:,\d{3})+
        |
        \d+
    )
    (?:\.\d+)?
    (?![A-Za-z])
    """,
    re.VERBOSE,
)


def normalize_text(text):
    text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("µ", "u").replace("μ", "u")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("−", "-")
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def parse_number(value):
    try:
        return float(str(value).strip().replace(",", ""))
    except (ValueError, TypeError):
        return None


def numbers_in_text(text):
    results = []
    for match in NUMBER_RE.finditer(text or ""):
        number = parse_number(match.group())
        if number is not None:
            results.append((number, match.start(), match.end()))
    return results


def is_reference_range(line):
    line = line.lower().strip()
    compact = line.replace(" ", "")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?-[-+]?\d+(?:\.\d+)?", compact):
        return True
    if (
        ("reference" in line or "biological reference" in line or "normal range" in line)
        and numbers_in_text(line)
    ):
        return True
    return False


def is_method_line(line):
    lowered = line.lower()
    keywords = [
        "method",
        "automated",
        "calculated",
        "microscopy",
        "flow cytometry",
        "optical",
        "immunoassay",
        "chemiluminescence",
        "photometry",
        "electrode",
        "enzymatic",
        "colorimetric",
    ]
    return any(keyword in lowered for keyword in keywords)


def is_header_line(line):
    lowered = line.lower().strip()
    headers = [
        "investigation",
        "result",
        "units",
        "unit",
        "reference interval",
        "reference range",
        "biological reference interval",
        "test name",
        "test",
    ]
    return lowered in headers or (
        "biological reference" in lowered and "interval" in lowered
    )


def clean_for_matching(text):
    text = (text or "").lower()
    replacements = {
        "haemoglobin": "hemoglobin",
        "haematocrit": "hematocrit",
        "leucocyte": "leukocyte",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def alias_matches_line(line, alias):
    normalized_line = clean_for_matching(line)
    normalized_alias = clean_for_matching(alias)
    if not normalized_alias:
        return False

    pattern = r"(?<![a-z])" + re.escape(normalized_alias) + r"(?![a-z])"
    return bool(re.search(pattern, normalized_line))


def find_test_in_line(line):
    matches = []
    for test_name, aliases in TEST_ALIASES.items():
        for alias in aliases:
            if alias_matches_line(line, alias):
                matches.append((test_name, len(clean_for_matching(alias))))
    if not matches:
        return None

    # Longest alias wins; this prevents generic "iron" / "sodium"
    # matches from beating more specific names.
    matches.sort(key=lambda item: item[1], reverse=True)
    return matches[0][0]


# ----------------------------------------------------------------
# Value plausibility
# ----------------------------------------------------------------
def is_probable_result_value(value, test_name, surrounding_text):
    if value is None or value < 0:
        return False

    limits = {
        "hemoglobin": (2, 30),
        "wbc": (100, 500000),
        "rbc": (0.1, 20),
        "platelets": (1000, 2000000),
        "hematocrit": (1, 100),
        "mcv": (20, 200),
        "mch": (5, 80),
        "mchc": (10, 60),
        "fasting_glucose": (20, 1000),
        "random_glucose": (20, 1000),
        "postprandial_glucose": (20, 1000),
        "hba1c": (1, 20),
        "alt": (0, 10000),
        "ast": (0, 10000),
        "alp": (0, 10000),
        "bilirubin_total": (0, 50),
        "bilirubin_direct": (0, 20),
        "albumin": (0.1, 10),
        "total_protein": (1, 20),
        "creatinine": (0.1, 30),
        "urea": (1, 500),
        "bun": (1, 300),
        "uric_acid": (0.1, 30),
        "total_cholesterol": (20, 1000),
        "ldl": (1, 500),
        "hdl": (1, 300),
        "triglycerides": (1, 5000),
        "tsh": (0.001, 1000),
        "free_t4": (0.01, 20),
        "free_t3": (0.01, 20),
        "sodium": (80, 200),
        "potassium": (1, 20),
        "calcium": (2, 20),
        "ferritin": (0, 10000),
        "serum_iron": (1, 1000),
    }

    if test_name in limits:
        low, high = limits[test_name]
        return low <= value <= high

    return True


# ----------------------------------------------------------------
# File extraction / OCR
# ----------------------------------------------------------------
def _ocr_available():
    """Return a clear dependency/runtime error if Tesseract cannot be executed."""
    if pytesseract is None or Image is None:
        raise ImportError(
            "OCR requires both pytesseract (Python package) and Pillow. "
            "Install them, and install the Tesseract OCR system executable separately."
        )
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise RuntimeError(
            "Tesseract OCR executable was not found or could not start. "
            "Install Tesseract (on Streamlit Community Cloud, add 'tesseract-ocr' "
            "to packages.txt; on Windows, install Tesseract and ensure it is on PATH)."
        ) from exc


def extract_text_from_file(file_path):
    """Extract text from supported reports; prefer PDF text before OCR fallback."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Report file not found: {path}")
    ext = path.suffix.lower()
    if ext not in {".txt", ".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
        raise ValueError(f"Unsupported file extension: {ext or '(none)'}")

    if path.stat().st_size == 0:
        raise ValueError("The uploaded report is empty.")

    if ext == ".txt":
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace") as file:
                text = file.read()
        except OSError as exc:
            raise OSError(f"Unable to read the TXT report: {exc}") from exc
        if not text.strip():
            raise ValueError("The TXT report contains no readable text.")
        return text

    if ext == ".pdf":
        if fitz is None:
            raise ImportError("PDF support requires PyMuPDF. Install the 'pymupdf' package.")
        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            raise ValueError(f"Unable to open this PDF; it may be damaged or password-protected: {exc}") from exc
        try:
            text_parts = [page.get_text("text") or "" for page in doc]
            text = "\n".join(text_parts).strip()
            # Some scanned PDFs contain tiny amounts of selectable text. Use OCR
            # when extracted text is effectively empty rather than relying on truthiness alone.
            if len(text) < 20:
                try:
                    _ocr_available()
                    ocr_parts = []
                    for page in doc:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                        page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        ocr_parts.append(pytesseract.image_to_string(page_image))
                    ocr_text = "\n".join(ocr_parts).strip()
                    if len(ocr_text) > len(text):
                        text = ocr_text
                except (ImportError, RuntimeError):
                    # Keep usable embedded text when OCR is unavailable; only
                    # fail if the PDF had no extractable text at all.
                    if not text.strip():
                        raise
            if not text.strip():
                raise ValueError("No text could be extracted from this PDF, even after OCR.")
            return text
        finally:
            doc.close()

    _ocr_available()
    try:
        with Image.open(path) as source_image:
            image = source_image.convert("RGB")
            text = pytesseract.image_to_string(image)
    except Exception as exc:
        raise ValueError(f"Unable to read/OCR this image report: {exc}") from exc
    if not text.strip():
        raise ValueError("OCR returned no readable text. Try a clearer, well-lit scan.")
    return text


# ----------------------------------------------------------------
# Value extraction
# ----------------------------------------------------------------
def extract_values(text):
    text = normalize_text(text)
    if not text.strip():
        return {}
    lines = text.splitlines()
    values = {}

    # Reports commonly put test name and value on different lines.
    SEARCH_AHEAD = 10

    for i, line in enumerate(lines):
        test_name = find_test_in_line(line)

        if test_name is None or test_name in values:
            continue

        candidates = [(line, "same_line")]

        for j in range(i + 1, min(len(lines), i + SEARCH_AHEAD + 1)):
            candidates.append((lines[j], "next_line"))

        # Prefer same-line values. If absent, inspect nearby lines.
        for candidate, source_type in candidates:
            candidate_clean = candidate.strip()

            if (
                not candidate_clean
                or is_header_line(candidate_clean)
                or is_method_line(candidate_clean)
                or is_reference_range(candidate_clean)
            ):
                continue

            num_candidates = numbers_in_text(candidate_clean)

            if not num_candidates:
                continue

            # Skip obvious reference-range lines and choose a plausible value.
            selected = None
            for number, _, _ in num_candidates:
                if test_name == "platelets" and (
                    "lakh" in candidate_clean.lower()
                    or "lakhs" in candidate_clean.lower()
                ):
                    number *= 100000

                if is_probable_result_value(number, test_name, candidate_clean):
                    selected = number
                    break

            if selected is not None:
                values[test_name] = selected
                break

    return values


# ----------------------------------------------------------------
# Status + explanations
# ----------------------------------------------------------------
def status_for(test_name, value):
    reference = REFERENCE_RANGES[test_name]

    if value < reference["min"]:
        return "LOW"
    if value > reference["max"]:
        return "HIGH"
    return "NORMAL"


def generate_dynamic_explanation(test_key, status, lang="en"):
    name = DISPLAY_NAMES.get(test_key, test_key.replace("_", " ").title())
    ref = REFERENCE_RANGES.get(
        test_key,
        {"min": 0, "max": 100, "unit": "units"},
    )

    if lang == "hi":
        direction = "अधिक" if status == "HIGH" else "कम"
        return {
            "name": name,
            "medical": f"{name} का स्तर निर्धारित संदर्भ सीमा ({ref['min']} - {ref['max']} {ref['unit']}) की तुलना में {direction} है।",
            "simple": f"💡 आसान भाषा में: {name} का मान सामान्य रूप से अपेक्षित सीमा से {direction} पाया गया है।",
            "conclusion": "📌 निष्कर्ष: इस परिणाम की व्याख्या अपने चिकित्सक के साथ करें।",
        }

    if lang == "mr":
        direction = "जास्त" if status == "HIGH" else "कमी"
        return {
            "name": name,
            "medical": f"{name} चे प्रमाण निर्धारित संदर्भ मर्यादा ({ref['min']} - {ref['max']} {ref['unit']}) पेक्षा {direction} आहे.",
            "simple": f"💡 सोप्या भाषेत: {name} ची पातळी अपेक्षित मर्यादेपेक्षा {direction} आहे.",
            "conclusion": "📌 निष्कर्ष: या निकालाचा अर्थ डॉक्टरांसोबत समजून घ्या.",
        }

    direction = "higher" if status == "HIGH" else "lower"
    return {
        "name": name,
        "medical": (
            f"{name} is {direction} than the configured reference range "
            f"({ref['min']} - {ref['max']} {ref['unit']})."
        ),
        "simple": (
            f"💡 What it means: the detected {name} value is "
            f"{direction} than this prototype's expected range."
        ),
        "conclusion": (
            "📌 Actionable conclusion: Discuss this result with a "
            "qualified healthcare professional."
        ),
    }


# ----------------------------------------------------------------
# Clinical pattern rules
# ----------------------------------------------------------------
def add_finding(findings, condition, reason, interpretation):
    findings.append(
        {
            "Condition": condition,
            "Reason": reason,
            "Interpretation": interpretation,
        }
    )


def detect_findings(values):
    findings = []

    # Diabetes-related patterns
    if "hba1c" in values:
        value = values["hba1c"]
        if value >= 6.5:
            add_finding(
                findings,
                "Diabetes Range Pattern",
                f"HbA1c = {value}%",
                "HbA1c is at or above the configured diabetes threshold; clinical confirmation is required.",
            )
        elif value >= 5.7:
            add_finding(
                findings,
                "Prediabetes Range Pattern",
                f"HbA1c = {value}%",
                "HbA1c is in the configured prediabetes threshold range.",
            )

    if "fasting_glucose" in values:
        value = values["fasting_glucose"]
        if value >= 126:
            add_finding(
                findings,
                "High Fasting Glucose",
                f"Fasting glucose = {value} mg/dL",
                "The value is at or above the configured 126 mg/dL threshold.",
            )
        elif value >= 100:
            add_finding(
                findings,
                "Impaired Fasting Glucose Pattern",
                f"Fasting glucose = {value} mg/dL",
                "The value is within the configured 100–125 mg/dL range.",
            )

    # Liver patterns
    if "alt" in values and values["alt"] > REFERENCE_RANGES["alt"]["max"]:
        add_finding(
            findings,
            "Elevated ALT Pattern",
            f"ALT = {values['alt']} U/L",
            "ALT is above the configured reference limit and should be interpreted clinically.",
        )

    if "ast" in values and values["ast"] > REFERENCE_RANGES["ast"]["max"]:
        add_finding(
            findings,
            "Elevated AST Pattern",
            f"AST = {values['ast']} U/L",
            "AST is above the configured reference limit and requires clinical context.",
        )

    # Kidney pattern
    if "creatinine" in values and values["creatinine"] > REFERENCE_RANGES["creatinine"]["max"]:
        add_finding(
            findings,
            "Elevated Serum Creatinine Pattern",
            f"Creatinine = {values['creatinine']} mg/dL",
            "Creatinine is above the configured reference limit and may warrant kidney-function review.",
        )

    # Anemia pattern
    if "hemoglobin" in values and values["hemoglobin"] < REFERENCE_RANGES["hemoglobin"]["min"]:
        add_finding(
            findings,
            "Low Hemoglobin Pattern",
            f"Hemoglobin = {values['hemoglobin']} g/dL",
            "Hemoglobin is below the configured reference limit; anemia is one possible clinical context.",
        )

    # Lipid pattern
    if "ldl" in values and values["ldl"] > REFERENCE_RANGES["ldl"]["max"]:
        add_finding(
            findings,
            "Elevated LDL Pattern",
            f"LDL = {values['ldl']} mg/dL",
            "LDL is above the simplified prototype reference limit.",
        )

    if "triglycerides" in values and values["triglycerides"] > REFERENCE_RANGES["triglycerides"]["max"]:
        add_finding(
            findings,
            "Elevated Triglyceride Pattern",
            f"Triglycerides = {values['triglycerides']} mg/dL",
            "Triglycerides are above the simplified prototype reference limit.",
        )

    return findings


# ----------------------------------------------------------------
# Public API used by app.py
# ----------------------------------------------------------------
def parse_and_analyze_report(file_path, lang="en"):
    raw_text = extract_text_from_file(file_path)
    extracted_values = extract_values(raw_text)

    results = []

    for test_key, value in extracted_values.items():
        if test_key not in REFERENCE_RANGES:
            continue

        reference = REFERENCE_RANGES[test_key]
        status = status_for(test_key, value)

        if (
            test_key in PREDEFINED_EXPLANATIONS
            and lang in PREDEFINED_EXPLANATIONS[test_key]
            and status in PREDEFINED_EXPLANATIONS[test_key][lang]
        ):
            explanation = PREDEFINED_EXPLANATIONS[test_key][lang][status]
            name = PREDEFINED_EXPLANATIONS[test_key][lang]["name"]
        else:
            explanation = generate_dynamic_explanation(
                test_key,
                status,
                lang,
            )
            name = explanation["name"]

        results.append(
            {
                "Test": name,
                "Value": str(value),
                "Unit": reference["unit"],
                "Reference Range": f"{reference['min']} - {reference['max']}",
                "Status": status,
                "MedicalExplanation": explanation["medical"],
                "SimpleExplanation": explanation["simple"],
                "Conclusion": explanation["conclusion"],
            }
        )

    return results


def get_report_dataframe(file_path):
    raw_text = extract_text_from_file(file_path)
    extracted_values = extract_values(raw_text)

    if pd is None:
        raise ImportError("pandas is required for dataframe generation.")

    rows = []

    for name, value in extracted_values.items():
        if name in REFERENCE_RANGES:
            reference = REFERENCE_RANGES[name]
            rows.append(
                {
                    "Test": DISPLAY_NAMES.get(name, name.replace("_", " ").title()),
                    "Value": value,
                    "Unit": reference["unit"],
                    "Reference Range": f"{reference['min']} - {reference['max']}",
                    "Status": status_for(name, value),
                }
            )

    return pd.DataFrame(rows)
