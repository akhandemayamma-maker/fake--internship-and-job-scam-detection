import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from PIL import Image
import pytesseract

import os
import re
from urllib.parse import quote_plus


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fake Job & Internship Scam Detector",
    page_icon="🔍",
    layout="wide"
)


# =========================================================
# TESSERACT OCR CONFIGURATION
# =========================================================

TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
]

tesseract_found = False

for path in TESSERACT_PATHS:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        tesseract_found = True
        break


# =========================================================
# TITLE
# =========================================================

st.title("🔍 Fake Job & Internship Scam Detector")

st.write(
    "Check whether a job or internship message looks like a "
    "possible scam using Machine Learning, OCR and verification tools."
)

st.divider()


# =========================================================
# LOAD DATASET
# =========================================================

try:
    data = pd.read_csv("dataset.csv")

except FileNotFoundError:
    st.error("❌ dataset.csv file not found.")
    st.info("Keep dataset.csv in the same folder as app.py.")
    st.stop()


# =========================================================
# CLEAN DATASET
# =========================================================

data.columns = data.columns.str.strip()

if "text" not in data.columns or "label" not in data.columns:
    st.error(
        "❌ dataset.csv must contain two columns: text and label"
    )
    st.stop()

data = data.dropna(subset=["text", "label"])

data["text"] = data["text"].astype(str)

data["label"] = (
    data["label"]
    .astype(str)
    .str.strip()
)


# =========================================================
# NORMALIZE LABELS
# =========================================================

data["label"] = data["label"].replace({
    "fake": "Fake",
    "Fake": "Fake",
    "FAKE": "Fake",

    "genuine": "Genuine",
    "Genuine": "Genuine",
    "GENUINE": "Genuine",

    "genuin": "Genuine",
    "Genuin": "Genuine"
})

data = data.drop_duplicates()


# =========================================================
# DATASET COUNTS
# =========================================================

total_records = len(data)

fake_count = (
    data["label"] == "Fake"
).sum()

genuine_count = (
    data["label"] == "Genuine"
).sum()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("📊 Dataset Information")

st.sidebar.metric(
    "Total Records",
    total_records
)

st.sidebar.metric(
    "Fake",
    fake_count
)

st.sidebar.metric(
    "Genuine",
    genuine_count
)

st.sidebar.divider()

if tesseract_found:
    st.sidebar.success(
        "✅ Tesseract OCR detected"
    )
else:
    st.sidebar.error(
        "❌ Tesseract OCR not detected"
    )


# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),

    (
        "classifier",
        MultinomialNB()
    )
])


model.fit(
    data["text"],
    data["label"]
)


# =========================================================
# SCAM KEYWORDS
# =========================================================

scam_keywords = [

    "registration fee",
    "processing fee",
    "training fee",
    "security deposit",
    "application fee",
    "joining fee",

    "pay",
    "payment",
    "deposit",

    "guaranteed job",
    "guaranteed selection",

    "bank account",
    "bank details",
    "account details",

    "debit card",
    "credit card",

    "otp",
    "password",
    "pin",

    "whatsapp",

    "no interview",
    "no experience",

    "huge salary",
    "high salary",

    "limited offer",
    "limited seats",

    "act now",
    "urgent",

    "refundable deposit",

    "work from home",

    "easy money"
]


# =========================================================
# FIND SCAM INDICATORS
# =========================================================

def find_scam_indicators(text):

    text_lower = text.lower()

    found = []

    for keyword in scam_keywords:

        if keyword in text_lower:
            found.append(keyword)

    return found


# =========================================================
# RISK SCORE
# =========================================================

def calculate_risk(prediction, indicators):

    count = len(indicators)

    if prediction == "Fake":

        if count >= 4:
            return 95

        elif count >= 3:
            return 90

        elif count >= 2:
            return 80

        elif count >= 1:
            return 70

        else:
            return 60

    else:

        if count >= 3:
            return 50

        elif count >= 1:
            return 25

        else:
            return 5


# =========================================================
# SCAM REASONS
# =========================================================

def get_reasons(indicators):

    reasons = []

    if "registration fee" in indicators:
        reasons.append(
            "⚠️ Registration fee is requested."
        )

    if "processing fee" in indicators:
        reasons.append(
            "⚠️ Processing fee is requested."
        )

    if "training fee" in indicators:
        reasons.append(
            "⚠️ Training fee is requested before employment."
        )

    if "security deposit" in indicators:
        reasons.append(
            "⚠️ Security deposit is requested."
        )

    if "application fee" in indicators:
        reasons.append(
            "⚠️ Application fee is requested."
        )

    if "joining fee" in indicators:
        reasons.append(
            "⚠️ Joining fee is requested."
        )

    if "guaranteed job" in indicators:
        reasons.append(
            "⚠️ Guaranteed job claim can be suspicious."
        )

    if "guaranteed selection" in indicators:
        reasons.append(
            "⚠️ Guaranteed selection is mentioned."
        )

    if (
        "bank account" in indicators
        or "bank details" in indicators
        or "account details" in indicators
    ):
        reasons.append(
            "⚠️ Bank information is requested."
        )

    if "debit card" in indicators:
        reasons.append(
            "⚠️ Debit card information is requested."
        )

    if "credit card" in indicators:
        reasons.append(
            "⚠️ Credit card information is requested."
        )

    if "otp" in indicators:
        reasons.append(
            "⚠️ OTP information is requested."
        )

    if "password" in indicators:
        reasons.append(
            "⚠️ Password information is requested."
        )

    if "pin" in indicators:
        reasons.append(
            "⚠️ PIN information is requested."
        )

    if "whatsapp" in indicators:
        reasons.append(
            "⚠️ Recruitment through WhatsApp requires extra verification."
        )

    if "huge salary" in indicators:
        reasons.append(
            "⚠️ Unrealistically high salary may be a warning sign."
        )

    if "high salary" in indicators:
        reasons.append(
            "⚠️ High-salary claim should be verified."
        )

    if "no experience" in indicators:
        reasons.append(
            "⚠️ No-experience claim with high benefits requires verification."
        )

    if (
        "limited offer" in indicators
        or "limited seats" in indicators
    ):
        reasons.append(
            "⚠️ Urgency or limited-seat pressure is mentioned."
        )

    if (
        "urgent" in indicators
        or "act now" in indicators
    ):
        reasons.append(
            "⚠️ Urgent action is requested."
        )

    if "work from home" in indicators:
        reasons.append(
            "ℹ️ Work-from-home claim detected; verify the employer."
        )

    if "easy money" in indicators:
        reasons.append(
            "⚠️ Easy-money claim may be suspicious."
        )

    return reasons


# =========================================================
# KNOWN COMPANIES
# =========================================================

KNOWN_COMPANIES = {

    "TCS": {
        "names": [
            "tcs",
            "tata consultancy services"
        ],
        "website": "https://www.tcs.com/",
        "careers": "https://www.tcs.com/careers"
    },

    "Infosys": {
        "names": [
            "infosys"
        ],
        "website": "https://www.infosys.com/",
        "careers": "https://www.infosys.com/careers/"
    },

    "Wipro": {
        "names": [
            "wipro"
        ],
        "website": "https://www.wipro.com/",
        "careers": "https://careers.wipro.com/"
    },

    "Accenture": {
        "names": [
            "accenture"
        ],
        "website": "https://www.accenture.com/",
        "careers": "https://www.accenture.com/in-en/careers"
    },

    "Cognizant": {
        "names": [
            "cognizant"
        ],
        "website": "https://www.cognizant.com/",
        "careers": "https://careers.cognizant.com/"
    },

    "Capgemini": {
        "names": [
            "capgemini"
        ],
        "website": "https://www.capgemini.com/",
        "careers": "https://www.capgemini.com/careers/"
    },

    "HCLTech": {
        "names": [
            "hcltech",
            "hcl technologies"
        ],
        "website": "https://www.hcltech.com/",
        "careers": "https://www.hcltech.com/careers"
    },

    "Tech Mahindra": {
        "names": [
            "tech mahindra"
        ],
        "website": "https://www.techmahindra.com/",
        "careers": "https://careers.techmahindra.com/"
    },

    "IBM": {
        "names": [
            "ibm"
        ],
        "website": "https://www.ibm.com/",
        "careers": "https://www.ibm.com/careers"
    },

    "Microsoft": {
        "names": [
            "microsoft"
        ],
        "website": "https://www.microsoft.com/",
        "careers": "https://careers.microsoft.com/"
    },

    "Google": {
        "names": [
            "google"
        ],
        "website": "https://www.google.com/",
        "careers": "https://careers.google.com/"
    },

    "Amazon": {
        "names": [
            "amazon"
        ],
        "website": "https://www.amazon.com/",
        "careers": "https://www.amazon.jobs/"
    },

    "Deloitte": {
        "names": [
            "deloitte"
        ],
        "website": "https://www.deloitte.com/",
        "careers": "https://www.deloitte.com/careers"
    },

    "EY": {
        "names": [
            "ey",
            "ernst young",
            "ernst & young"
        ],
        "website": "https://www.ey.com/",
        "careers": "https://www.ey.com/en_in/careers"
    },

    "KPMG": {
        "names": [
            "kpmg"
        ],
        "website": "https://kpmg.com/",
        "careers": "https://kpmg.com/in/en/careers.html"
    },

    "PwC": {
        "names": [
            "pwc",
            "pricewaterhousecoopers"
        ],
        "website": "https://www.pwc.com/",
        "careers": "https://www.pwc.com/gx/en/careers.html"
    },

    "Oracle": {
        "names": [
            "oracle"
        ],
        "website": "https://www.oracle.com/",
        "careers": "https://www.oracle.com/careers/"
    },

    "SAP": {
        "names": [
            "sap"
        ],
        "website": "https://www.sap.com/",
        "careers": "https://www.sap.com/about/careers.html"
    },

    "Honeywell": {
        "names": [
            "honeywell"
        ],
        "website": "https://www.honeywell.com/",
        "careers": "https://careers.honeywell.com/"
    },

    "Bosch": {
        "names": [
            "bosch"
        ],
        "website": "https://www.bosch.com/",
        "careers": "https://www.bosch.com/careers/"
    },

    "Zoho": {
        "names": [
            "zoho"
        ],
        "website": "https://www.zoho.com/",
        "careers": "https://www.zoho.com/careers/"
    },

    "Freshworks": {
        "names": [
            "freshworks"
        ],
        "website": "https://www.freshworks.com/",
        "careers": "https://www.freshworks.com/company/careers/"
    },

    "Cisco": {
        "names": [
            "cisco"
        ],
        "website": "https://www.cisco.com/",
        "careers": "https://jobs.cisco.com/"
    },

    "Intel": {
        "names": [
            "intel"
        ],
        "website": "https://www.intel.com/",
        "careers": "https://jobs.intel.com/"
    },

    "Adobe": {
        "names": [
            "adobe"
        ],
        "website": "https://www.adobe.com/",
        "careers": "https://www.adobe.com/careers.html"
    },

    "ServiceNow": {
        "names": [
            "servicenow"
        ],
        "website": "https://www.servicenow.com/",
        "careers": "https://careers.servicenow.com/"
    },

    "Flipkart": {
        "names": [
            "flipkart"
        ],
        "website": "https://www.flipkart.com/",
        "careers": "https://www.flipkartcareers.com/"
    },

    "Walmart": {
        "names": [
            "walmart"
        ],
        "website": "https://www.walmart.com/",
        "careers": "https://careers.walmart.com/"
    }
}


# =========================================================
# COMPANY NAME DETECTION
# =========================================================

def detect_company_name(text):

    if not text:
        return None

    text_lower = text.lower()

    # -----------------------------------------------------
    # 1. Check known companies
    # -----------------------------------------------------

    for company, info in KNOWN_COMPANIES.items():

        for name in info["names"]:

            if re.search(
                r"\b" + re.escape(name) + r"\b",
                text_lower
            ):
                return company

    # -----------------------------------------------------
    # 2. Company name after common labels
    # -----------------------------------------------------

    patterns = [

        r"company\s*name\s*[:\-]\s*([A-Za-z][A-Za-z0-9 &.,'-]{2,60})",

        r"company\s*[:\-]\s*([A-Za-z][A-Za-z0-9 &.,'-]{2,60})",

        r"organization\s*name\s*[:\-]\s*([A-Za-z][A-Za-z0-9 &.,'-]{2,60})",

        r"employer\s*[:\-]\s*([A-Za-z][A-Za-z0-9 &.,'-]{2,60})",

        r"hiring\s+(?:at|for)\s+([A-Za-z][A-Za-z0-9 &.,'-]{2,60})",

        r"careers\s+at\s+([A-Za-z][A-Za-z0-9 &.,'-]{2,60})"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            company = match.group(1).strip()

            company = re.split(
                r"\n|\.|,?\s+(?:is|are|offers|hiring|careers|jobs)\b",
                company,
                flags=re.IGNORECASE
            )[0].strip()

            if len(company) >= 3:
                return company

    # -----------------------------------------------------
    # 3. Detect company from email domain
    # -----------------------------------------------------

    emails = re.findall(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    for email in emails:

        domain = email.split("@")[-1].lower()

        domain_name = domain.split(".")[0]

        ignored_domains = {
            "gmail",
            "yahoo",
            "hotmail",
            "outlook",
            "rediffmail",
            "protonmail"
        }

        if domain_name not in ignored_domains:

            company = domain_name.replace(
                "-",
                " "
            ).replace(
                "_",
                " "
            )

            return company.title()

    return None


# =========================================================
# EXTRACT EMAILS
# =========================================================

def extract_emails(text):

    return list(
        dict.fromkeys(
            re.findall(
                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                text
            )
        )
    )


# =========================================================
# EXTRACT PHONE NUMBERS
# =========================================================

def extract_phone_numbers(text):

    numbers = re.findall(
        r"(?:\+91[\s-]?)?[6-9]\d{9}",
        text
    )

    return list(
        dict.fromkeys(numbers)
    )


# =========================================================
# EXTRACT URLS
# =========================================================

def extract_urls(text):

    urls = re.findall(
        r"(?:https?://|www\.)[^\s<>\"]+",
        text,
        flags=re.IGNORECASE
    )

    cleaned = []

    for url in urls:

        url = url.rstrip(
            ".,;:)]}"
        )

        if url not in cleaned:
            cleaned.append(url)

    return cleaned


# =========================================================
# COMPANY VERIFICATION - AUTOMATIC ANALYSIS
# =========================================================

def get_company_info(company):
    """Return known-company information when available."""
    if not company:
        return None

    company_lower = company.strip().lower()

    for known_name, info in KNOWN_COMPANIES.items():
        if known_name.lower() == company_lower:
            return info

        for name in info.get("names", []):
            if name.lower() == company_lower:
                return info

    return None


def extract_domains_from_urls(urls):
    """Extract clean domains from detected URLs."""
    domains = []

    for url in urls:
        clean_url = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
        clean_url = re.sub(r"^www\.", "", clean_url, flags=re.IGNORECASE)
        domain = clean_url.split("/")[0].split(":")[0].lower()

        if domain and domain not in domains:
            domains.append(domain)

    return domains


def expected_company_domains(company):
    """Create likely official email/website domains for a known company."""
    info = get_company_info(company)
    domains = []

    if info:
        website = info.get("website", "")
        website_match = re.search(r"https?://(?:www\.)?([^/]+)", website)
        if website_match:
            domains.append(website_match.group(1).lower())

    # Useful fallback for companies detected from an email domain.
    simple_name = re.sub(r"[^a-z0-9]", "", company.lower())
    if simple_name:
        domains.append(simple_name + ".com")

    return list(dict.fromkeys(domains))


def verify_email_domains(company, emails):
    """Check whether extracted email addresses use a public/personal domain."""
    public_domains = {
        "gmail.com",
        "yahoo.com",
        "yahoo.co.in",
        "hotmail.com",
        "outlook.com",
        "rediffmail.com",
        "protonmail.com",
        "icloud.com",
        "aol.com",
        "live.com",
    }

    expected = expected_company_domains(company)
    results = []

    for email in emails:
        domain = email.split("@")[-1].lower().strip()

        if domain in public_domains:
            results.append({
                "email": email,
                "status": "warning",
                "message": "Personal/public email domain detected."
            })
        elif expected and any(domain == item or domain.endswith("." + item) for item in expected):
            results.append({
                "email": email,
                "status": "success",
                "message": "Email domain matches the detected company domain."
            })
        else:
            results.append({
                "email": email,
                "status": "info",
                "message": "Email domain could not be confirmed automatically."
            })

    return results


def verify_websites(company, urls):
    """Compare extracted website domains with the detected company."""
    domains = extract_domains_from_urls(urls)
    expected = expected_company_domains(company)

    if not domains:
        return {
            "status": "warning",
            "message": "No website URL was detected in the uploaded text/image.",
            "domains": []
        }

    if expected:
        for domain in domains:
            if any(domain == item or domain.endswith("." + item) for item in expected):
                return {
                    "status": "success",
                    "message": "Detected website domain matches the company domain.",
                    "domains": domains
                }

        return {
            "status": "warning",
            "message": "Detected website domain does not match the expected company domain.",
            "domains": domains
        }

    return {
        "status": "info",
        "message": "Website was detected, but automatic domain matching is unavailable for this company.",
        "domains": domains
    }


def build_verification_flags(company, text, emails, phones, urls, indicators):
    """Build a compact automatic verification summary."""
    flags = []

    if company:
        flags.append(("success", "Company name detected from the uploaded content."))
    else:
        flags.append(("warning", "Company name could not be detected automatically."))

    if emails:
        email_checks = verify_email_domains(company or "", emails)
        if any(item["status"] == "warning" for item in email_checks):
            flags.append(("warning", "Personal/public recruiter email detected."))
        elif any(item["status"] == "success" for item in email_checks):
            flags.append(("success", "Recruiter email domain matches the detected company."))
        else:
            flags.append(("info", "Recruiter email was found but could not be fully verified."))
    else:
        flags.append(("warning", "No recruiter email was detected."))

    website_check = verify_websites(company or "", urls)
    flags.append((website_check["status"], website_check["message"]))

    if phones:
        flags.append(("success", "Phone number detected in the uploaded content."))
    else:
        flags.append(("info", "No phone number detected."))

    if indicators:
        flags.append(("warning", f"{len(indicators)} scam-related warning indicator(s) detected."))
    else:
        flags.append(("success", "No major scam keywords detected."))

    return flags


def show_company_verification(text):
    """Display automatic company, email, website and red-flag verification."""
    detected_company = detect_company_name(text)

    st.subheader("🏢 Company Detection & Verification")

    if not detected_company:
        st.warning("⚠️ Company name could not be detected automatically.")
        st.info("Try uploading a clearer image containing the company name.")

        manual_company = st.text_input(
            "🏢 Enter company name manually",
            key="manual_company_name"
        )

        if manual_company.strip():
            detected_company = manual_company.strip()
        else:
            return

    st.success(f"🏢 Detected Company: {detected_company}")

    emails = extract_emails(text)
    phones = extract_phone_numbers(text)
    urls = extract_urls(text)
    indicators = find_scam_indicators(text)

    # ---------------------------------------------------------
    # Extracted details
    # ---------------------------------------------------------
    st.markdown("### 📋 Extracted Details")

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        st.write("**🏢 Company:**")
        st.write(detected_company)

        st.write("**📧 Email(s):**")
        if emails:
            for email in emails:
                st.write(f"• {email}")
        else:
            st.write("No email detected.")

    with detail_col2:
        st.write("**📞 Phone Number(s):**")
        if phones:
            for phone in phones:
                st.write(f"• {phone}")
        else:
            st.write("No phone number detected.")

        st.write("**🌐 URL(s):**")
        if urls:
            for url in urls:
                st.write(f"• {url}")
        else:
            st.write("No URL detected.")

    # ---------------------------------------------------------
    # Email verification
    # ---------------------------------------------------------
    st.markdown("### 🔐 Email Domain Verification")

    if emails:
        email_checks = verify_email_domains(detected_company, emails)

        for check in email_checks:
            if check["status"] == "success":
                st.success(f"✅ {check['email']} — {check['message']}")
            elif check["status"] == "warning":
                st.warning(f"⚠️ {check['email']} — {check['message']}")
            else:
                st.info(f"ℹ️ {check['email']} — {check['message']}")

        expected_domains = expected_company_domains(detected_company)
        if expected_domains:
            st.caption("Expected company domain: " + ", ".join("@" + d for d in expected_domains))
    else:
        st.warning("⚠️ No recruiter email was detected.")

    # ---------------------------------------------------------
    # Website verification
    # ---------------------------------------------------------
    st.markdown("### 🌐 Website Verification")

    website_check = verify_websites(detected_company, urls)

    if website_check["status"] == "success":
        st.success("✅ " + website_check["message"])
    elif website_check["status"] == "warning":
        st.warning("⚠️ " + website_check["message"])
    else:
        st.info("ℹ️ " + website_check["message"])

    if website_check["domains"]:
        st.write("**Detected domain(s):** " + ", ".join(website_check["domains"]))

    # ---------------------------------------------------------
    # Automatic red-flag summary
    # ---------------------------------------------------------
    st.markdown("### 🛡️ Automatic Verification Summary")

    flags = build_verification_flags(
        detected_company,
        text,
        emails,
        phones,
        urls,
        indicators
    )

    for status, message in flags:
        if status == "success":
            st.success("✅ " + message)
        elif status == "warning":
            st.warning("⚠️ " + message)
        else:
            st.info("ℹ️ " + message)

    # ---------------------------------------------------------
    # Detected red flags
    # ---------------------------------------------------------
    st.markdown("### 🚩 Red Flags Found")

    if indicators:
        for indicator in indicators:
            st.write(f"🔸 {indicator.title()}")
    else:
        st.success("✅ No major scam keywords were detected in this content.")

    # ---------------------------------------------------------
    # Final verification status
    # ---------------------------------------------------------
    warning_count = sum(1 for status, _ in flags if status == "warning")

    st.markdown("### 📊 Verification Status")

    if warning_count >= 3:
        st.error("🔴 HIGH ATTENTION REQUIRED — Multiple verification warnings found.")
    elif warning_count >= 1:
        st.warning("🟠 VERIFY CAREFULLY — One or more verification warnings found.")
    else:
        st.success("🟢 NO AUTOMATIC VERIFICATION WARNING FOUND")

    st.caption(
        "Note: Automatic checks are only screening signals. They do not prove that a job, company or certificate is genuine."
    )

    # ---------------------------------------------------------
    # Company-specific checklist
    # ---------------------------------------------------------
    st.markdown("### 📝 Final Verification Checklist")

    checklist = [
        "Company website matches the offer",
        "Recruiter email uses the company's official domain",
        "Job/internship can be verified through an official careers page",
        "No registration or processing fee is requested",
        "No security deposit or training fee is requested",
        "No OTP, PIN or password is requested",
        "Bank/card details are not requested without proper verification",
    ]

    for index, item in enumerate(checklist):
        st.checkbox(item, key=f"company_verify_{index}")

    # No Google / LinkedIn / Reviews / Internship search buttons are used here.


# =========================================================
# CERTIFICATE VERIFICATION
# =========================================================

def show_certificate_verification():

    st.subheader(
        "📜 Certificate Verification"
    )

    certificate_name = st.text_input(
        "Enter Certificate / Organization / Course Name",
        placeholder="Example: TCS iON Certificate"
    )

    if certificate_name.strip():

        certificate = certificate_name.strip()

        google_certificate = (
            "https://www.google.com/search?q="
            + quote_plus(
                certificate +
                " certificate verification"
            )
        )

        official_certificate = (
            "https://www.google.com/search?q="
            + quote_plus(
                certificate +
                " official website"
            )
        )

        issuer_search = (
            "https://www.google.com/search?q="
            + quote_plus(
                certificate +
                " issuing organization"
            )
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.link_button(
                "🔍 Google Search",
                google_certificate,
                use_container_width=True
            )

        with col2:

            st.link_button(
                "🌐 Official Website",
                official_certificate,
                use_container_width=True
            )

        with col3:

            st.link_button(
                "✅ Certificate Verification",
                issuer_search,
                use_container_width=True
            )

        st.info(
            "⚠️ Verify certificates only through the issuing "
            "organization or its official website."
        )


# =========================================================
# SAFETY CHECKLIST
# =========================================================

def show_safety_checklist():

    st.subheader(
        "🛡️ Job / Internship Safety Checklist"
    )

    checklist = [

        "Verify the company through its official website.",

        "Check whether the recruiter uses an official company email.",

        "Check the company's official careers page.",

        "Do not pay registration, training or security-deposit fees.",

        "Never share OTP, PIN or password.",

        "Never share banking credentials without proper verification.",

        "Verify certificates directly with the issuing organization.",

        "Be careful with urgent or guaranteed-job offers."
    ]

    for item in checklist:

        st.checkbox(
            item,
            value=False,
            key="safe_" + str(
                checklist.index(item)
            )
        )


# =========================================================
# DISPLAY ML RESULT
# =========================================================

def display_result(user_text):

    if not user_text.strip():

        st.warning(
            "⚠️ No text available for analysis."
        )

        return

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = model.predict(
        [user_text]
    )[0]

    probabilities = model.predict_proba(
        [user_text]
    )[0]

    confidence = max(
        probabilities
    ) * 100

    # -----------------------------------------------------
    # Scam indicators
    # -----------------------------------------------------

    indicators = find_scam_indicators(
        user_text
    )

    # -----------------------------------------------------
    # Risk
    # -----------------------------------------------------

    risk_score = calculate_risk(
        prediction,
        indicators
    )

    reasons = get_reasons(
        indicators
    )

    st.divider()

    # -----------------------------------------------------
    # Main result
    # -----------------------------------------------------

    if prediction == "Fake":

        st.error(
            "🚨 RESULT: FAKE / POSSIBLE SCAM"
        )

    else:

        st.success(
            "✅ RESULT: GENUINE / LOWER RISK"
        )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Prediction",
            prediction
        )

    with col2:

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

    with col3:

        st.metric(
            "Risk Score",
            f"{risk_score}%"
        )

    # -----------------------------------------------------
    # Risk analysis
    # -----------------------------------------------------

    st.subheader(
        "📈 Risk Analysis"
    )

    st.progress(
        min(risk_score, 100) / 100
    )

    if risk_score >= 70:

        st.error(
            f"🔴 HIGH RISK - {risk_score}%"
        )

    elif risk_score >= 40:

        st.warning(
            f"🟠 MEDIUM RISK - {risk_score}%"
        )

    else:

        st.success(
            f"🟢 LOW RISK - {risk_score}%"
        )

    # -----------------------------------------------------
    # Scam indicators
    # -----------------------------------------------------

    st.subheader(
        "🚩 Detected Scam Indicators"
    )

    if indicators:

        for indicator in indicators:

            st.write(
                f"🔸 {indicator.title()}"
            )

    else:

        st.info(
            "No major scam keywords detected."
        )

    # -----------------------------------------------------
    # Scam reasons
    # -----------------------------------------------------

    st.subheader(
        "💡 Scam Reasons"
    )

    if reasons:

        for reason in reasons:

            st.warning(
                reason
            )

    else:

        st.success(
            "No major warning signs were detected from the entered text."
        )

    # -----------------------------------------------------
    # Download report
    # -----------------------------------------------------

    report = f"""
FAKE JOB & INTERNSHIP SCAM DETECTOR
===================================

Input:
{user_text}

Prediction:
{prediction}

Model Confidence:
{confidence:.2f}%

Risk Score:
{risk_score}%

Detected Indicators:
{", ".join(indicators) if indicators else "None"}

Reasons:
{"; ".join(reasons) if reasons else "No major warning signs detected."}

Company Detected:
{detect_company_name(user_text) or "Not detected"}

Emails:
{", ".join(extract_emails(user_text)) or "None"}

Phone Numbers:
{", ".join(extract_phone_numbers(user_text)) or "None"}

URLs:
{", ".join(extract_urls(user_text)) or "None"}

NOTE:
This application is an ML-based initial screening tool.
Always verify the company and job offer through official sources.
Never share OTP, passwords, PINs or sensitive banking information.
"""

    st.download_button(
        label="📥 Download Detection Report",
        data=report,
        file_name="scam_detection_report.txt",
        mime="text/plain"
    )


# =========================================================
# MAIN TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "📝 Text Input",
    "🖼️ Image Upload",
    "📜 Certificate Check"
])


# =========================================================
# TAB 1 - TEXT INPUT
# =========================================================

with tab1:

    st.subheader(
        "📝 Check Job / Internship Message"
    )

    user_text = st.text_area(
        "Paste the job or internship message here",
        height=200,
        placeholder=(
            "Example:\n"
            "Earn ₹50,000 per week from home.\n"
            "Pay ₹2,000 registration fee.\n"
            "Contact us on WhatsApp."
        ),
        key="text_input"
    )

    check_button = st.button(
        "🔍 Check Message",
        type="primary",
        key="text_check_button"
    )

    if check_button:

        if user_text.strip() == "":

            st.warning(
                "⚠️ Please enter a job or internship message."
            )

        else:

            display_result(
                user_text
            )

            st.divider()

            show_company_verification(
                user_text
            )

            st.divider()

            show_safety_checklist()


# =========================================================
# TAB 2 - IMAGE UPLOAD
# =========================================================

with tab2:

    st.subheader(
        "🖼️ Upload Job / Internship Screenshot"
    )

    st.write(
        "Upload a clear screenshot, job advertisement, "
        "offer letter or certificate image."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "png",
            "jpg",
            "jpeg"
        ],
        key="job_image"
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        analyze_button = st.button(
            "🔍 Analyze Image",
            type="primary",
            key="image_check_button"
        )

        if analyze_button:

            if not tesseract_found:

                st.error(
                    "❌ Tesseract OCR was not found."
                )

                st.info(
                    "Install Tesseract OCR and restart Streamlit."
                )

            else:

                try:

                    # -------------------------------------
                    # OCR
                    # -------------------------------------

                    image_rgb = image.convert(
                        "RGB"
                    )

                    extracted_text = (
                        pytesseract.image_to_string(
                            image_rgb
                        )
                    )

                    if not extracted_text.strip():

                        st.warning(
                            "⚠️ No readable text was found."
                        )

                        st.info(
                            "Please upload a clear image "
                            "with readable text."
                        )

                    else:

                        # ---------------------------------
                        # OCR output
                        # ---------------------------------

                        st.subheader(
                            "📄 OCR Extracted Text"
                        )

                        st.text_area(
                            "Detected Text",
                            extracted_text,
                            height=220,
                            key="ocr_output"
                        )

                        # ---------------------------------
                        # ML analysis
                        # ---------------------------------

                        display_result(
                            extracted_text
                        )

                        st.divider()

                        # ---------------------------------
                        # Company detection
                        # ---------------------------------

                        show_company_verification(
                            extracted_text
                        )

                        st.divider()

                        # ---------------------------------
                        # Safety checklist
                        # ---------------------------------

                        show_safety_checklist()

                except Exception as e:

                    st.error(
                        "❌ OCR could not process the image."
                    )

                    st.write(
                        f"Error: {e}"
                    )


# =========================================================
# TAB 3 - CERTIFICATE VERIFICATION
# =========================================================

with tab3:

    show_certificate_verification()


# =========================================================
# DATASET ANALYSIS
# =========================================================

st.divider()

st.subheader(
    "📊 Dataset Analysis"
)

chart_data = pd.DataFrame({
    "Type": [
        "Fake",
        "Genuine"
    ],

    "Count": [
        fake_count,
        genuine_count
    ]
})

st.bar_chart(
    chart_data.set_index(
        "Type"
    )
)

col1, col2 = st.columns(2)

with col1:

    st.info(
        f"🚨 Fake Records: {fake_count}"
    )

with col2:

    st.success(
        f"✅ Genuine Records: {genuine_count}"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "⚠️ This application is an ML-based screening tool. "
    "Always verify job offers through the company's official "
    "website and never share OTP, passwords, PINs or sensitive "
    "banking information."
)