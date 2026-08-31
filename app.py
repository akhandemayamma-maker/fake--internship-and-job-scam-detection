import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Fake Job Scam Detector",
    page_icon="🔍",
    layout="wide"
)


# --------------------------------------------------
# SAMPLE TRAINING DATA
# --------------------------------------------------

training_data = {

    "text": [

        # Fake examples
        "Pay registration fee to get job",
        "Pay money and get guaranteed job",
        "Send money to confirm your internship",
        "Work from home earn huge salary no interview",
        "Pay security deposit before joining",
        "You are selected pay processing fee",
        "Send OTP and bank details to receive salary",
        "Limited seats pay registration charges",
        "Guaranteed job without interview",
        "Pay training fee to get internship",

        # Legitimate examples
        "Software developer internship with technical interview",
        "Company is hiring software engineers through official website",
        "Candidates will attend an interview before selection",
        "Internship opportunity for computer science students",
        "Apply through the official company career portal",
        "Selected candidates will receive an offer letter",
        "Campus recruitment drive for engineering students",
        "Software testing internship with interview process",
        "Data analyst internship with skill assessment",
        "Web developer position at a registered company"
    ],

    "label": [

        "Fake", "Fake", "Fake", "Fake", "Fake",
        "Fake", "Fake", "Fake", "Fake", "Fake",

        "Legitimate", "Legitimate", "Legitimate",
        "Legitimate", "Legitimate", "Legitimate",
        "Legitimate", "Legitimate", "Legitimate",
        "Legitimate"
    ]
}


data = pd.DataFrame(training_data)


# --------------------------------------------------
# MACHINE LEARNING MODEL
# --------------------------------------------------

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

model.fit(data["text"], data["label"])


# --------------------------------------------------
# SCAM KEYWORDS
# --------------------------------------------------

scam_keywords = [

    "registration fee",
    "processing fee",
    "security deposit",
    "pay money",
    "pay fee",
    "send money",
    "guaranteed job",
    "guaranteed salary",
    "no interview",
    "otp",
    "bank details",
    "limited seats",
    "urgent payment",
    "training fee"
]


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🔍 Fake Internship & Job Scam Detector")

st.write(
    "An AI-based system to identify potentially suspicious "
    "job and internship offers."
)

st.divider()


# --------------------------------------------------
# INPUT
# --------------------------------------------------

job_text = st.text_area(
    "📄 Enter Job / Internship Description",
    placeholder=(
        "Example: Pay ₹5000 registration fee and get "
        "a guaranteed work-from-home job..."
    ),
    height=180
)


# --------------------------------------------------
# CHECK BUTTON
# --------------------------------------------------

if st.button("🚀 Analyze Job", use_container_width=True):

    if not job_text.strip():

        st.warning("⚠️ Please enter a job or internship description.")

    else:

        text = job_text.lower()


        # ------------------------------------------
        # ML PREDICTION
        # ------------------------------------------

        prediction = model.predict([job_text])[0]

        probabilities = model.predict_proba([job_text])[0]

        classes = model.classes_

        probability_dict = dict(
            zip(classes, probabilities)
        )


        # ------------------------------------------
        # KEYWORD ANALYSIS
        # ------------------------------------------

        detected_keywords = []

        for keyword in scam_keywords:

            if keyword in text:

                detected_keywords.append(keyword)


        # ------------------------------------------
        # RISK SCORE
        # ------------------------------------------

        ml_fake_probability = (
            probability_dict.get("Fake", 0) * 100
        )

        keyword_score = min(
            len(detected_keywords) * 7,
            35
        )

        risk_score = min(
            int(ml_fake_probability * 0.65 + keyword_score),
            100
        )


        # ------------------------------------------
        # RESULT
        # ------------------------------------------

        st.subheader("📊 Analysis Result")


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Risk Score",
                f"{risk_score}/100"
            )


        with col2:

            st.metric(
                "ML Prediction",
                prediction
            )


        with col3:

            st.metric(
                "Suspicious Keywords",
                len(detected_keywords)
            )


        # ------------------------------------------
        # FINAL DECISION
        # ------------------------------------------

        if risk_score >= 60:

            st.error(
                "🚨 HIGH RISK: Possible Fake Job / Internship"
            )

        elif risk_score >= 35:

            st.warning(
                "⚠️ MEDIUM RISK: Verify this opportunity carefully"
            )

        else:

            st.success(
                "✅ LOW RISK: Appears to be legitimate"
            )


        # ------------------------------------------
        # SUSPICIOUS KEYWORDS
        # ------------------------------------------

        if detected_keywords:

            st.subheader("🚨 Suspicious Indicators")

            for keyword in detected_keywords:

                st.write(
                    f"• **{keyword}**"
                )

        else:

            st.info(
                "No major scam keywords detected."
            )


        # ------------------------------------------
        # CONFIDENCE
        # ------------------------------------------

        st.subheader("🤖 AI Confidence")

        st.progress(
            int(
                probability_dict.get("Fake", 0) * 100
            )
        )

        st.write(
            f"Fake: "
            f"{probability_dict.get('Fake', 0) * 100:.2f}%"
        )

        st.write(
            f"Legitimate: "
            f"{probability_dict.get('Legitimate', 0) * 100:.2f}%"
        )


        # ------------------------------------------
        # SAFETY RECOMMENDATIONS
        # ------------------------------------------

        st.subheader("🛡️ Safety Recommendations")

        if risk_score >= 60:

            st.write("❌ Do not pay any registration or processing fee.")

            st.write("❌ Do not share OTP, PIN or banking passwords.")

            st.write("❌ Verify the company through its official website.")

            st.write("❌ Do not click suspicious job links.")

        else:

            st.write(
                "✅ Verify the company and recruiter before applying."
            )

            st.write(
                "✅ Check whether the job is listed on the official company website."
            )

            st.write(
                "✅ Never share OTP or banking passwords."
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "⚠️ This system provides an AI-based prediction. "
    "Always verify job opportunities through official sources."
)