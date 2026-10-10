import re
import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import fitz
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pytesseract

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from bs4 import BeautifulSoup
import fitz


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake Job & Internship Scam Detector",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# SESSION STATE
# ============================================================

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

if "source_name" not in st.session_state:
    st.session_state.source_name = ""

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# ============================================================
# TITLE
# ============================================================

st.title("🔍 Fake Internship & Job Scam Detector")

st.markdown(
    """
Analyze **job descriptions, job advertisement images, PDFs, or job links**
using Machine Learning and scam-indicator analysis.

> ⚠️ This system provides risk signals only.  
> A low-risk result does **not** guarantee that a job is genuine.
"""
)


# ============================================================
# TEST TESSERACT
# ============================================================

with st.sidebar:
    st.header("⚙️ System Status")

    if st.button("🔧 Test OCR"):
        try:
            version = pytesseract.get_tesseract_version()
            st.success(f"✅ Tesseract working\n\nVersion: {version}")
        except Exception as e:
            st.error("❌ Tesseract not found.")
            st.code(str(e))

    st.caption("Tesseract path:")
    st.code(TESSERACT_PATH)


# ============================================================
# TRAINING DATA
# ============================================================

suspicious_examples = [
    "Earn 50000 per week from home. Pay 2000 registration fee.",
    "Congratulations you are selected. Pay registration fee immediately.",
    "Work from home and earn huge income. No interview required.",
    "Send your bank account details to receive salary.",
    "Pay security deposit before joining the company.",
    "Limited vacancies. Apply immediately by paying a processing fee.",
    "You have been selected for an internship. Pay certificate fee.",
    "Earn 1 lakh per month without experience. Contact on WhatsApp.",
    "Urgent hiring. Pay refundable security amount.",
    "Send OTP and bank details for verification.",
    "Guaranteed job with no interview. Registration fee required.",
    "Pay training fee before receiving offer letter.",
    "Work from home earning guaranteed 50000 weekly.",
    "WhatsApp us immediately to confirm your job.",
    "Your job is reserved. Pay money today.",
    "No qualification required. Earn huge money instantly.",
    "Provide ATM card details for salary processing.",
    "Pay 5000 to activate your employee account.",
    "Immediate joining after payment of processing charges.",
    "Send Aadhaar, PAN and bank details through WhatsApp.",
    "Selected candidates must pay verification fee.",
    "Earn unlimited income with zero investment after paying registration.",
    "Job offer expires today. Pay now.",
    "Remote job with guaranteed salary and no interview.",
    "Pay refundable deposit before interview.",
    "Company requires security money before joining.",
    "Contact recruiter only through Telegram.",
    "Pay document verification fee immediately.",
    "Guaranteed internship after paying registration charges."
]

genuine_examples = [
    "Software developer internship for final year students. Apply through official company careers portal.",
    "Internship selection includes technical interview and coding assessment.",
    "Candidates will receive a monthly stipend based on company policy.",
    "Apply through the official company website with your resume.",
    "Software engineer position requiring Java Python and SQL skills.",
    "Candidates must attend an online technical interview.",
    "Internship duration is three months with certificate upon successful completion.",
    "Selected candidates will receive joining information by official email.",
    "Applicants should submit resume through the company career portal.",
    "Job location Bengaluru. Salary will be discussed during the interview.",
    "Graduate software engineer role for computer science students.",
    "Candidates should have knowledge of programming and database systems.",
    "Internship applications are accepted through the official careers page.",
    "The recruitment process consists of aptitude test technical interview and HR round.",
    "Company provides mentorship and project experience during internship.",
    "Freshers can apply through the official recruitment portal.",
    "Candidates will be contacted by the recruitment team for interview scheduling.",
    "Work location and compensation will be discussed during the selection process.",
    "Applicants should meet the eligibility criteria mentioned in the job description.",
    "The company does not charge candidates for recruitment."
]


# Expand training examples
texts = suspicious_examples * 15 + genuine_examples * 15
labels = ["Suspicious"] * (len(suspicious_examples) * 15) + \
         ["Genuine"] * (len(genuine_examples) * 15)


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000
        )
    ),
    (
        "classifier",
        MultinomialNB()
    )
])

model.fit(texts, labels)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = str(text)

    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# ============================================================
# OCR NORMALIZATION
# ============================================================

def normalize_ocr_text(text):

    if not text:
        return ""

    text = text.replace("\x00", " ")

    text = text.replace("RS.", "Rs.")
    text = text.replace("rs.", "Rs.")
    text = text.replace("R5.", "Rs.")

    text = text.replace("—", "-")
    text = text.replace("–", "-")

    # Common OCR mistakes
    text = text.replace("permonth", "per month")
    text = text.replace("/month", " per month")

    text = re.sub(r"₹\s+", "₹", text)

    return clean_text(text)


# ============================================================
# CHECK TESSERACT
# ============================================================

def check_tesseract():

    try:
        version = pytesseract.get_tesseract_version()
        return True, str(version)

    except Exception as e:
        return False, str(e)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = ImageOps.exif_transpose(image)

    # Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Increase size for OCR
    width, height = image.size

    if width < 1600:
        scale = 1600 / width
        image = image.resize(
            (int(width * scale), int(height * scale))
        )

    # Grayscale
    gray = ImageOps.grayscale(image)

    # Contrast
    gray = ImageEnhance.Contrast(gray).enhance(2.0)

    # Sharpness
    gray = ImageEnhance.Sharpness(gray).enhance(2.0)

    # Slight smoothing
    gray = gray.filter(ImageFilter.SHARPEN)

    return gray


# ============================================================
# OCR IMAGE
# ============================================================

def extract_text_from_image(uploaded_file):

    try:

        image = Image.open(uploaded_file)

        processed_image = preprocess_image(image)

        text = pytesseract.image_to_string(
            processed_image,
            config="--oem 3 --psm 6"
        )

        return normalize_ocr_text(text)

    except Exception as e:

        st.error(f"❌ Image OCR Error: {e}")

        return ""


# ============================================================
# PDF TEXT + OCR
# ============================================================

def extract_text_from_pdf(uploaded_file):

    try:

        pdf_bytes = uploaded_file.read()

        pdf = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        all_text = []

        total_pages = len(pdf)

        if total_pages == 0:
            pdf.close()
            return ""

        progress = st.progress(0)

        for page_number in range(total_pages):

            page = pdf[page_number]

            # ------------------------------------------------
            # FIRST TRY NORMAL PDF TEXT EXTRACTION
            # ------------------------------------------------

            normal_text = page.get_text("text")

            normal_text = clean_text(normal_text)

            if len(normal_text.strip()) >= 30:

                all_text.append(normal_text)

            else:

                # ------------------------------------------------
                # SCANNED PDF -> OCR
                # ------------------------------------------------

                pix = page.get_pixmap(
                    matrix=fitz.Matrix(3, 3),
                    alpha=False
                )

                image = Image.frombytes(
                    "RGB",
                    [pix.width, pix.height],
                    pix.samples
                )

                image = preprocess_image(image)

                ocr_text = pytesseract.image_to_string(
                    image,
                    config="--oem 3 --psm 6"
                )

                ocr_text = normalize_ocr_text(ocr_text)

                if ocr_text:
                    all_text.append(ocr_text)

            progress.progress(
                int(((page_number + 1) / total_pages) * 100)
            )

        progress.empty()

        pdf.close()

        final_text = "\n\n".join(all_text)

        final_text = normalize_ocr_text(final_text)

        return final_text.strip()

    except Exception as e:

        st.error(f"❌ PDF extraction/OCR error: {e}")

        return ""


# ============================================================
# WEBPAGE EXTRACTION
# ============================================================

def extract_text_from_url(url):

    try:

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for element in soup(
            ["script", "style", "noscript", "svg"]
        ):
            element.decompose()

        text = soup.get_text(
            separator="\n"
        )

        return clean_text(text)

    except Exception as e:

        st.error(f"❌ Website extraction error: {e}")

        return ""


# ============================================================
# EMAIL EXTRACTION
# ============================================================

def extract_emails(text):

    if not text:
        return []

    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

    emails = re.findall(
        pattern,
        text
    )

    return list(dict.fromkeys(emails))


# ============================================================
# PHONE EXTRACTION
# ============================================================

def extract_phones(text):

    if not text:
        return []

    pattern = r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"

    phones = re.findall(
        pattern,
        text
    )

    return list(dict.fromkeys(phones))


# ============================================================
# WEBSITE EXTRACTION
# ============================================================

def extract_websites(text):

    if not text:
        return []

    pattern = r"(?:https?://|www\.)[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s]*)?"

    websites = re.findall(
        pattern,
        text,
        flags=re.IGNORECASE
    )

    return list(dict.fromkeys(websites))


# ============================================================
# SALARY / STIPEND EXTRACTION
# ============================================================

def extract_salary(text):

    if not text:
        return []

    normalized = normalize_ocr_text(text)

    results = []

    patterns = [

        r"₹\s*[\d,]+\s*-\s*₹?\s*[\d,]+\s*(?:per|/)?\s*month",

        r"₹\s*[\d,]+\s*to\s*₹?\s*[\d,]+\s*(?:per|/)?\s*month",

        r"Rs\.?\s*[\d,]+\s*-\s*Rs\.?\s*[\d,]+\s*(?:per|/)?\s*month",

        r"\b\d[\d,]*\s*-\s*\d[\d,]*\s*(?:per|/)?\s*month",

        r"(?:salary|stipend)\s*[:\-]?\s*"
        r"(?:₹|Rs\.?|%)?\s*[\d,]+\s*"
        r"(?:-|to)?\s*"
        r"(?:₹|Rs\.?|%)?\s*[\d,]*\s*"
        r"(?:per|/)?\s*month",

        r"₹\s*[\d,]+\s*(?:per|/)?\s*month",

        r"Rs\.?\s*[\d,]+\s*(?:per|/)?\s*month"
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            normalized,
            flags=re.IGNORECASE
        )

        results.extend(matches)

    cleaned = []

    for item in results:

        item = re.sub(
            r"\s+",
            " ",
            item.strip()
        )

        if item and item not in cleaned:
            cleaned.append(item)

    return cleaned


# ============================================================
# COMPANY EXTRACTION
# ============================================================

def extract_company(text):

    if not text:
        return "Not detected"

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # --------------------------------------------------------
    # Company: XYZ
    # --------------------------------------------------------

    patterns = [
        r"(?:company|organization|employer|hiring company)"
        r"\s*[:\-]\s*([A-Za-z0-9&.,'()\- ]{3,80})",

        r"(?:company name|organization name)"
        r"\s*[:\-]\s*([A-Za-z0-9&.,'()\- ]{3,80})"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = match.group(1).strip()

            value = re.sub(
                r"\s+",
                " ",
                value
            )

            if len(value) > 2:
                return value


    # --------------------------------------------------------
    # Company before website
    # --------------------------------------------------------

    for line in lines:

        if "website" in line.lower():

            parts = re.split(
                r"[:|]",
                line,
                maxsplit=1
            )

            if len(parts) == 2:

                left = parts[0].strip()

                left = re.sub(
                    r"(?i)website",
                    "",
                    left
                ).strip()

                if len(left) >= 3:
                    return left


    # --------------------------------------------------------
    # Company-style names
    # --------------------------------------------------------

    company_keywords = [
        "solutions",
        "technologies",
        "technology",
        "systems",
        "software",
        "consulting",
        "services",
        "private limited",
        "pvt ltd",
        "limited",
        "labs",
        "industries"
    ]

    for line in lines:

        low = line.lower()

        if any(
            keyword in low
            for keyword in company_keywords
        ):

            if len(line) <= 100:

                bad_words = [
                    "job description",
                    "requirements",
                    "responsibilities",
                    "skills",
                    "salary",
                    "benefits",
                    "internship duration"
                ]

                if not any(
                    word in low
                    for word in bad_words
                ):
                    return line


    # --------------------------------------------------------
    # Email domain fallback
    # --------------------------------------------------------

    emails = extract_emails(text)

    if emails:

        email = emails[0]

        domain = email.split("@")[-1]

        if domain:

            company_name = domain.split(".")[0]

            if company_name.lower() not in [
                "gmail",
                "yahoo",
                "outlook",
                "hotmail",
                "rediffmail"
            ]:

                return company_name.replace(
                    "-",
                    " "
                ).title()


    return "Not detected"


# ============================================================
# JOB TITLE EXTRACTION
# ============================================================

def extract_job_title(text):

    if not text:
        return "Not detected"

    patterns = [

        r"(?:job title|position|role|designation)"
        r"\s*[:\-]\s*([^\n]{3,100})",

        r"(?:hiring for|looking for)"
        r"\s*[:\-]?\s*([^\n]{3,100})"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = match.group(1).strip()

            value = re.sub(
                r"\s+",
                " ",
                value
            )

            if len(value) <= 100:
                return value


    # Search common job titles
    common_titles = [
        "Software Developer",
        "Software Engineer",
        "Software Development Intern",
        "Python Developer",
        "Java Developer",
        "Web Developer",
        "Frontend Developer",
        "Backend Developer",
        "Full Stack Developer",
        "Data Analyst",
        "Data Scientist",
        "Machine Learning Intern",
        "AI Intern",
        "Cyber Security Intern",
        "Intern",
        "Graduate Engineer Trainee"
    ]

    for title in common_titles:

        if title.lower() in text.lower():
            return title

    return "Not detected"


# ============================================================
# LOCATION EXTRACTION
# ============================================================

def extract_location(text):

    if not text:
        return "Not detected"

    patterns = [
        r"(?:location|job location|work location)"
        r"\s*[:\-]\s*([A-Za-z ,.-]{2,80})"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = match.group(1).strip()

            return value

    common_locations = [
        "Bengaluru",
        "Bangalore",
        "Hyderabad",
        "Chennai",
        "Pune",
        "Mumbai",
        "Delhi",
        "Noida",
        "Gurugram",
        "Mysuru",
        "Mysore",
        "Raichur"
    ]

    for location in common_locations:

        if location.lower() in text.lower():
            return location

    return "Not detected"


# ============================================================
# SCAM INDICATORS
# ============================================================

SCAM_INDICATORS = {

    "Payment Request": [
        "registration fee",
        "processing fee",
        "security deposit",
        "pay fee",
        "pay money",
        "payment required",
        "joining fee",
        "activation fee",
        "verification fee",
        "training fee",
        "deposit"
    ],

    "Sensitive Information Request": [
        "otp",
        "bank account",
        "bank details",
        "atm card",
        "cvv",
        "password",
        "pin",
        "aadhaar",
        "pan card",
        "send your documents"
    ],

    "Suspicious Promise": [
        "guaranteed job",
        "guaranteed income",
        "earn 50000",
        "earn 1 lakh",
        "huge salary",
        "no experience required",
        "no interview",
        "unlimited income",
        "easy money"
    ],

    "Urgency / Pressure": [
        "apply immediately",
        "urgent",
        "limited vacancies",
        "today only",
        "pay now",
        "immediately",
        "expires today",
        "act now"
    ],

    "Suspicious Recruitment": [
        "contact on whatsapp",
        "contact on telegram",
        "telegram",
        "whatsapp only",
        "selected without interview",
        "job reserved"
    ]
}


def detect_scam_indicators(text):

    if not text:
        return {}

    text_lower = text.lower()

    detected = {}

    for category, keywords in SCAM_INDICATORS.items():

        found = []

        for keyword in keywords:

            if keyword.lower() in text_lower:
                found.append(keyword)

        if found:
            detected[category] = list(
                dict.fromkeys(found)
            )

    return detected


# ============================================================
# ML PREDICTION
# ============================================================

def predict_job(text):

    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]

    classes = model.classes_

    probability_dict = dict(
        zip(classes, probabilities)
    )

    suspicious_probability = probability_dict.get(
        "Suspicious",
        0
    ) * 100

    genuine_probability = probability_dict.get(
        "Genuine",
        0
    ) * 100

    return (
        prediction,
        suspicious_probability,
        genuine_probability
    )


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk_score(
    text,
    suspicious_probability,
    indicators
):

    score = 0

    # ML component
    score += suspicious_probability * 0.45

    # Indicator component
    indicator_count = sum(
        len(values)
        for values in indicators.values()
    )

    score += min(
        indicator_count * 7,
        35
    )

    # Additional high-risk words
    high_risk_words = [
        "registration fee",
        "security deposit",
        "otp",
        "cvv",
        "atm card",
        "guaranteed job",
        "pay now"
    ]

    text_lower = text.lower()

    for word in high_risk_words:

        if word in text_lower:
            score += 5

    return min(
        int(round(score)),
        100
    )


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(score):

    if score >= 70:
        return "HIGH RISK"

    elif score >= 40:
        return "MEDIUM RISK"

    else:
        return "LOW RISK"


# ============================================================
# EXPLANATION
# ============================================================

def generate_explanation(
    score,
    indicators,
    suspicious_probability
):

    explanations = []

    if suspicious_probability >= 60:

        explanations.append(
            "The machine-learning model detected several patterns "
            "similar to suspicious job advertisements."
        )

    if "Payment Request" in indicators:

        explanations.append(
            "The advertisement appears to request money or fees "
            "from the applicant."
        )

    if "Sensitive Information Request" in indicators:

        explanations.append(
            "The advertisement contains requests for sensitive "
            "personal or financial information."
        )

    if "Urgency / Pressure" in indicators:

        explanations.append(
            "Urgent or pressure-based language was detected."
        )

    if "Suspicious Promise" in indicators:

        explanations.append(
            "The advertisement contains unusually strong "
            "income or job guarantees."
        )

    if "Suspicious Recruitment" in indicators:

        explanations.append(
            "Unusual recruitment/contact methods were detected."
        )

    if not explanations:

        explanations.append(
            "No major predefined scam indicators were detected. "
            "Still verify the employer independently."
        )

    return explanations


# ============================================================
# REPORT
# ============================================================

def generate_report(
    text,
    source_name,
    company,
    job_title,
    salary,
    location,
    emails,
    phones,
    websites,
    prediction,
    suspicious_probability,
    genuine_probability,
    risk_score,
    risk_level,
    indicators
):

    report = []

    report.append(
        "FAKE INTERNSHIP & JOB SCAM DETECTOR"
    )

    report.append("=" * 50)

    report.append(
        f"Source: {source_name}"
    )

    report.append(
        f"Company: {company}"
    )

    report.append(
        f"Job Title: {job_title}"
    )

    report.append(
        f"Salary/Stipend: {', '.join(salary) if salary else 'Not detected'}"
    )

    report.append(
        f"Location: {location}"
    )

    report.append(
        f"Email: {', '.join(emails) if emails else 'Not detected'}"
    )

    report.append(
        f"Phone: {', '.join(phones) if phones else 'Not detected'}"
    )

    report.append(
        f"Website: {', '.join(websites) if websites else 'Not detected'}"
    )

    report.append("")
    report.append("ANALYSIS")
    report.append("-" * 50)

    report.append(
        f"ML Prediction: {prediction}"
    )

    report.append(
        f"Suspicious Probability: {suspicious_probability:.2f}%"
    )

    report.append(
        f"Genuine Probability: {genuine_probability:.2f}%"
    )

    report.append(
        f"Risk Score: {risk_score}/100"
    )

    report.append(
        f"Risk Level: {risk_level}"
    )

    report.append("")
    report.append("SCAM INDICATORS")
    report.append("-" * 50)

    if indicators:

        for category, values in indicators.items():

            report.append(
                f"{category}: {', '.join(values)}"
            )

    else:

        report.append(
            "No predefined scam indicators detected."
        )

    report.append("")
    report.append("SAFETY RECOMMENDATIONS")
    report.append("-" * 50)

    report.append(
        "1. Do not pay registration or recruitment fees."
    )

    report.append(
        "2. Do not share OTP, CVV, PIN or banking passwords."
    )

    report.append(
        "3. Verify the company through its official website."
    )

    report.append(
        "4. Check whether the recruiter uses an official company email."
    )

    report.append(
        "5. Do not trust guaranteed-job or guaranteed-income claims."
    )

    report.append("")
    report.append("EXTRACTED TEXT")
    report.append("-" * 50)

    report.append(text)

    return "\n".join(report)


# ============================================================
# INPUT METHOD
# ============================================================

input_method = st.radio(
    "📥 Choose Job Input Method",
    [
        "Job Description",
        "Job Advertisement Image",
        "Job Advertisement PDF",
        "Job Link"
    ],
    horizontal=True
)


# ============================================================
# INPUT: JOB DESCRIPTION
# ============================================================

if input_method == "Job Description":

    st.subheader("📝 Enter Job Advertisement")

    job_text = st.text_area(
        "Paste the complete job description here:",
        height=300,
        placeholder=(
            "Example:\n"
            "Software Developer Intern\n"
            "Company: ABC Technologies\n"
            "Location: Bengaluru\n"
            "Apply through official website..."
        )
    )

    if st.button(
        "🔍 Analyze Job Description",
        type="primary"
    ):

        st.session_state.extracted_text = ""
        st.session_state.source_name = ""
        st.session_state.analysis_done = False

        if len(job_text.strip()) < 20:

            st.error(
                "❌ Please enter at least 20 characters."
            )

        else:

            st.session_state.extracted_text = clean_text(
                job_text
            )

            st.session_state.source_name = (
                "Job Description"
            )

            st.session_state.analysis_done = True


# ============================================================
# INPUT: IMAGE
# ============================================================

elif input_method == "Job Advertisement Image":

    st.subheader("🖼️ Upload Job Advertisement Image")

    uploaded_image = st.file_uploader(
        "Upload JPG / JPEG / PNG image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_image is not None:

        st.image(
            uploaded_image,
            caption="Uploaded Job Advertisement",
            use_container_width=True
        )

        if st.button(
            "🔍 Extract Text & Analyze Image",
            type="primary"
        ):

            st.session_state.extracted_text = ""
            st.session_state.source_name = ""
            st.session_state.analysis_done = False

            ok, version = check_tesseract()

            if not ok:

                st.error(
                    "❌ Tesseract OCR is not available."
                )

                st.code(
                    TESSERACT_PATH
                )

            else:

                with st.spinner(
                    "Reading image and performing OCR..."
                ):

                    extracted_text = (
                        extract_text_from_image(
                            uploaded_image
                        )
                    )

                if (
                    extracted_text
                    and len(extracted_text.strip()) >= 20
                ):

                    st.session_state.extracted_text = (
                        extracted_text
                    )

                    st.session_state.source_name = (
                        "Job Advertisement Image"
                    )

                    st.session_state.analysis_done = True

                    st.success(
                        "✅ Image text extracted successfully!"
                    )

                    st.text_area(
                        "OCR Text:",
                        extracted_text,
                        height=300
                    )

                else:

                    st.session_state.extracted_text = ""

                    st.error(
                        "❌ Could not extract enough readable text from the image."
                    )


# ============================================================
# INPUT: PDF
# ============================================================

elif input_method == "Job Advertisement PDF":

    st.subheader("📄 Upload Job Advertisement PDF")

    uploaded_pdf = st.file_uploader(
        "Upload Job Advertisement PDF",
        type=["pdf"],
        key="pdf_upload"
    )

    if uploaded_pdf is not None:

        st.success(
            f"PDF uploaded successfully — "
            f"{uploaded_pdf.size / 1024:.1f} KB"
        )

        if st.button(
            "📄 Extract Text & Analyze PDF",
            type="primary"
        ):

            # IMPORTANT:
            # Clear previous result before processing
            st.session_state.extracted_text = ""
            st.session_state.source_name = ""
            st.session_state.analysis_done = False

            ok, version = check_tesseract()

            if not ok:

                st.error(
                    "❌ Tesseract OCR is not available."
                )

                st.info(
                    "Check that this file exists:"
                )

                st.code(
                    TESSERACT_PATH
                )

            else:

                with st.spinner(
                    "Reading PDF pages and performing OCR. "
                    "Please wait..."
                ):

                    extracted_text = (
                        extract_text_from_pdf(
                            uploaded_pdf
                        )
                    )

                if (
                    extracted_text
                    and len(extracted_text.strip()) >= 20
                ):

                    st.session_state.extracted_text = (
                        extracted_text
                    )

                    st.session_state.source_name = (
                        "Job Advertisement PDF"
                    )

                    st.session_state.analysis_done = True

                    st.success(
                        "✅ PDF text extracted successfully!"
                    )

                    st.text_area(
                        "PDF / OCR Text:",
                        extracted_text,
                        height=350
                    )

                else:

                    # IMPORTANT:
                    # Do not leave previous result
                    st.session_state.extracted_text = ""
                    st.session_state.source_name = ""
                    st.session_state.analysis_done = False

                    st.error(
                        "❌ No readable text was found in this PDF."
                    )

                    st.warning(
                        "The PDF may be image-based, "
                        "low quality, password protected, "
                        "or contain handwritten/very unclear text."
                    )


# ============================================================
# INPUT: JOB LINK
# ============================================================

elif input_method == "Job Link":

    st.subheader("🔗 Enter Job Advertisement Link")

    job_url = st.text_input(
        "Job advertisement URL:",
        placeholder="https://example.com/job"
    )

    if st.button(
        "🌐 Extract & Analyze Job Link",
        type="primary"
    ):

        st.session_state.extracted_text = ""
        st.session_state.source_name = ""
        st.session_state.analysis_done = False

        if not job_url.strip():

            st.error(
                "❌ Please enter a job URL."
            )

        else:

            with st.spinner(
                "Opening job webpage..."
            ):

                extracted_text = (
                    extract_text_from_url(
                        job_url.strip()
                    )
                )

            if (
                extracted_text
                and len(extracted_text.strip()) >= 20
            ):

                st.session_state.extracted_text = (
                    extracted_text
                )

                st.session_state.source_name = (
                    "Job Link"
                )

                st.session_state.analysis_done = True

                st.success(
                    "✅ Job webpage text extracted successfully!"
                )

            else:

                st.error(
                    "❌ Could not extract readable text from this webpage."
                )


# ============================================================
# ANALYSIS
# ============================================================

if (
    st.session_state.get("analysis_done")
    and st.session_state.get("extracted_text")
    and len(
        st.session_state.extracted_text.strip()
    ) >= 20
):

    text = st.session_state.extracted_text

    st.divider()

    st.header("📊 Job Analysis Result")

    # --------------------------------------------------------
    # Extract information
    # --------------------------------------------------------

    company = extract_company(text)

    job_title = extract_job_title(text)

    salary = extract_salary(text)

    location = extract_location(text)

    emails = extract_emails(text)

    phones = extract_phones(text)

    websites = extract_websites(text)

    indicators = detect_scam_indicators(text)

    # --------------------------------------------------------
    # ML
    # --------------------------------------------------------

    prediction, suspicious_probability, genuine_probability = (
        predict_job(text)
    )

    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    risk_score = calculate_risk_score(
        text,
        suspicious_probability,
        indicators
    )

    risk_level = get_risk_level(
        risk_score
    )

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    st.subheader("🏢 Job Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.write(
            f"**Company:** {company}"
        )

        st.write(
            f"**Job Title:** {job_title}"
        )

        st.write(
            f"**Location:** {location}"
        )

    with info_col2:

        st.write(
            "**Salary/Stipend:** "
            + (
                ", ".join(salary)
                if salary
                else "Not detected"
            )
        )

        st.write(
            "**Email:** "
            + (
                ", ".join(emails)
                if emails
                else "Not detected"
            )
        )

        st.write(
            "**Website:** "
            + (
                ", ".join(websites)
                if websites
                else "Not detected"
            )
        )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    st.divider()

    st.subheader("🚨 Risk Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Risk Score",
            f"{risk_score}/100"
        )

    with col2:

        st.metric(
            "ML Suspicious",
            f"{suspicious_probability:.1f}%"
        )

    with col3:

        st.metric(
            "ML Genuine",
            f"{genuine_probability:.1f}%"
        )

    if risk_level == "HIGH RISK":

        st.error(
            "🔴 HIGH RISK"
        )

    elif risk_level == "MEDIUM RISK":

        st.warning(
            "🟠 MEDIUM RISK"
        )

    else:

        st.success(
            "🟢 LOW RISK"
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    st.write(
        f"**Machine Learning Prediction:** "
        f"{prediction}"
    )

    # --------------------------------------------------------
    # Scam indicators
    # --------------------------------------------------------

    st.divider()

    st.subheader("🚩 Scam Indicators")

    if indicators:

        for category, values in indicators.items():

            st.warning(
                f"**{category}:** "
                + ", ".join(values)
            )

    else:

        st.success(
            "✅ No predefined scam indicators detected."
        )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    st.divider()

    st.subheader("💡 Why this result?")

    explanations = generate_explanation(
        risk_score,
        indicators,
        suspicious_probability
    )

    for explanation in explanations:

        st.write(
            "• " + explanation
        )

    # --------------------------------------------------------
    # Contact information
    # --------------------------------------------------------

    st.divider()

    st.subheader("📞 Contact Information")

    if emails:

        st.write(
            "**Email:** "
            + ", ".join(emails)
        )

    else:

        st.write(
            "**Email:** Not detected"
        )

    if phones:

        st.write(
            "**Phone:** "
            + ", ".join(phones)
        )

    else:

        st.write(
            "**Phone:** Not detected"
        )

    if websites:

        st.write(
            "**Website:** "
            + ", ".join(websites)
        )

    else:

        st.write(
            "**Website:** Not detected"
        )

    # --------------------------------------------------------
    # Safety recommendations
    # --------------------------------------------------------

    st.divider()

    st.subheader("🛡️ Safety Recommendations")

    recommendations = [
        "Do not pay registration or recruitment fees.",
        "Never share OTP, CVV, PIN or banking passwords.",
        "Verify the company through its official website.",
        "Check whether the recruiter uses an official company email.",
        "Be careful with guaranteed-job or guaranteed-income claims.",
        "Verify the company and recruiter independently before accepting an offer."
    ]

    for recommendation in recommendations:

        st.write(
            "✅ " + recommendation
        )

    # --------------------------------------------------------
    # Extracted text
    # --------------------------------------------------------

    st.divider()

    with st.expander("📄 View Extracted Text"):

        st.text_area(
            "Extracted Content",
            text,
            height=400
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = generate_report(
        text=text,
        source_name=st.session_state.source_name,
        company=company,
        job_title=job_title,
        salary=salary,
        location=location,
        emails=emails,
        phones=phones,
        websites=websites,
        prediction=prediction,
        suspicious_probability=suspicious_probability,
        genuine_probability=genuine_probability,
        risk_score=risk_score,
        risk_level=risk_level,
        indicators=indicators
    )

    st.download_button(
        label="📥 Download Analysis Report",
        data=report,
        file_name="job_scam_analysis_report.txt",
        mime="text/plain"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fake Internship & Job Scam Detector | "
    "Python + Streamlit + TF-IDF + Multinomial Naive Bayes + OCR"
)

st.caption(
    "⚠️ This is an academic/demo system. "
    "ML predictions and risk scores should not be treated as "
    "proof that a job is genuine or fraudulent."
)
