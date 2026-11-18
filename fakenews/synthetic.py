"""Phase 1: synthetic fake/real news generator (English, Hinglish, Hindi).

THE DATA IS SYNTHETIC. It exists to build, test and demonstrate the pipeline. Scores on it are not
model performance. No real people are named; institutions are generic.

Each row records *how* it was made (template_id, case_type, style, noisy_label) so evaluators can see
exactly what signal exists in the data and audit what the models learned.
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .utils import FakeNewsError, ensure_dir, file_sha256, log, write_json

TOPICS = ["health", "govt_schemes", "economy", "technology", "weather", "education", "elections", "transport"]

# --------------------------------------------------------------------------------------------
# Slot values. Both classes draw from the same lists, so slot words carry no label signal.
# --------------------------------------------------------------------------------------------
SLOTS = {
    "en": {
        "city": ["Patna", "Lucknow", "Jaipur", "Pune", "Bhopal", "Indore", "Nagpur", "Kochi", "Guwahati", "Ranchi",
                 "Surat", "Varanasi", "Dehradun", "Raipur", "Mysuru", "Agra", "Kanpur", "Nashik", "Madurai", "Vijayawada"],
        "state": ["Bihar", "Uttar Pradesh", "Rajasthan", "Maharashtra", "Madhya Pradesh", "Kerala", "Assam",
                  "Jharkhand", "Gujarat", "Odisha", "Karnataka", "Tamil Nadu"],
        "disease": ["dengue", "diabetes", "malaria", "kidney stones", "cancer", "typhoid", "asthma",
                    "high blood pressure", "jaundice", "the flu"],
        "remedy": ["garlic water", "turmeric milk", "neem leaves", "lemon with hot water", "onion juice",
                   "papaya leaf juice", "clove oil", "ginger tea"],
        "item": ["laptops", "smartphones", "bicycles", "solar panels", "scooters", "tablets", "gas cylinders"],
        "group": ["students", "farmers", "women", "senior citizens", "unemployed youth", "daily wage workers"],
        "amount": ["₹2,000", "₹5,000", "₹10,000", "₹15,000", "₹50,000"],
        "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "month": ["January", "February", "March", "April", "May", "June", "July", "August", "September",
                  "October", "November", "December"],
    },
    "hinglish": {
        "city": ["Patna", "Lucknow", "Jaipur", "Pune", "Bhopal", "Indore", "Nagpur", "Ranchi", "Varanasi", "Kanpur",
                 "Agra", "Raipur", "Surat", "Dehradun"],
        "state": ["Bihar", "UP", "Rajasthan", "Maharashtra", "MP", "Jharkhand", "Gujarat", "Odisha", "Haryana"],
        "disease": ["dengue", "sugar", "malaria", "pathri", "cancer", "BP", "piliya", "typhoid"],
        "remedy": ["lehsun ka paani", "haldi wala doodh", "neem ke patte", "nimbu paani", "adrak ki chai",
                   "pyaaz ka ras"],
        "item": ["laptop", "smartphone", "cycle", "scooty", "solar panel", "tablet"],
        "group": ["students", "kisaanon", "mahilaon", "buzurgon", "berozgar yuvaon", "mazdooron"],
        "amount": ["₹2,000", "₹5,000", "₹10,000", "₹15,000"],
        "day": ["Somvaar", "Mangalvaar", "Budhvaar", "Guruvaar", "Shukravaar", "Shanivaar", "Ravivaar"],
        "month": ["January", "February", "March", "April", "May", "June", "July", "August", "September",
                  "October", "November", "December"],
    },
    "hi": {
        "city": ["पटना", "लखनऊ", "जयपुर", "भोपाल", "इंदौर", "नागपुर", "रांची", "वाराणसी", "देहरादून", "रायपुर",
                 "कानपुर", "आगरा"],
        "state": ["बिहार", "उत्तर प्रदेश", "राजस्थान", "मध्य प्रदेश", "महाराष्ट्र", "झारखंड", "गुजरात", "ओडिशा",
                  "हरियाणा"],
        "disease": ["डेंगू", "मधुमेह", "मलेरिया", "पथरी", "कैंसर", "पीलिया", "टाइफाइड"],
        "remedy": ["लहसुन का पानी", "हल्दी वाला दूध", "नीम की पत्तियां", "नींबू पानी", "अदरक की चाय"],
        "item": ["लैपटॉप", "स्मार्टफोन", "साइकिल", "स्कूटी", "सोलर पैनल", "टैबलेट"],
        "group": ["छात्रों", "किसानों", "महिलाओं", "बुजुर्गों", "बेरोजगार युवाओं", "मजदूरों"],
        "amount": ["₹2,000", "₹5,000", "₹10,000", "₹15,000"],
        "day": ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"],
        "month": ["जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर",
                  "नवंबर", "दिसंबर"],
    },
}
NUMERIC_SLOTS = {
    "n": lambda r: str(r.randint(10, 90)),
    "n_small": lambda r: str(r.randint(2, 7)),
    "coin": lambda r: str(r.choice([75, 100, 125, 150, 200])),
    "small_amt": lambda r: str(r.choice([1, 2, 5, 10])),
    "date": lambda r: str(r.randint(1, 28)),
}

# --------------------------------------------------------------------------------------------
# Templates: (label, topic, case_type, text). TEMPLATES holds the real ones, FAKE_TEMPLATES the fake ones.
#   label: real | fake
#   case_type: standard | surprising_true | debunk_report (hard real) | calm_fake (hard fake)
# --------------------------------------------------------------------------------------------
TEMPLATES = {
    "en": [
        ("real", "weather", "standard", "Schools in {city} will remain closed on {day} due to heavy rain, the district administration said on its official website."),
        ("real", "weather", "standard", "The weather department has forecast moderate to heavy rainfall in parts of {state} over the next {n_small} days."),
        ("real", "weather", "standard", "The weather department said the monsoon in {state} is likely to be normal this year, with rainfall close to the long-period average."),
        ("real", "transport", "standard", "The railways will run {n} additional trains during the festive season to manage the rush, according to an official release."),
        ("real", "transport", "standard", "The {city} metro will extend its operating hours on weekends from {month}, the metro corporation announced."),
        ("real", "economy", "standard", "The central bank kept the repo rate unchanged at its policy meeting on {day}, citing stable inflation."),
        ("real", "economy", "surprising_true", "The central bank will issue a commemorative ₹{coin} coin to mark the anniversary of the constitution, an official notification said."),
        ("real", "health", "standard", "The district health department in {city} has started a free vaccination drive for children under five, officials said."),
        ("real", "health", "debunk_report", "The Health Ministry clarified that a viral message claiming {remedy} cures {disease} is false and urged people to consult doctors."),
        ("real", "technology", "standard", "The telecom regulator has asked operators to improve call quality in {state} after receiving {n} complaints, a statement said."),
        ("real", "technology", "debunk_report", "The telecom department said a viral claim that 5G towers spread {disease} in {city} is false and has no scientific basis."),
        ("real", "education", "standard", "The {state} education board will declare class 12 results in the last week of {month}, the board said in a notice."),
        ("real", "education", "standard", "The state government has extended the last date for scholarship applications for {group} to {month} {date}, officials said."),
        ("real", "elections", "standard", "The election commission announced that voting in {state} will be held in {n_small} phases, with counting on a single day."),
        ("real", "govt_schemes", "standard", "The state government said {n} lakh {group} have received benefits under its pension scheme this year, according to official data."),
        ("real", "technology", "surprising_true", "The space agency successfully placed a weather satellite into orbit on {day}, the agency said in a statement."),
        ("real", "elections", "debunk_report", "The election commission said a viral video of voting machines being tampered with in {city} is old and unrelated to the current polls."),
        ("real", "govt_schemes", "debunk_report", "Officials said a viral message offering free {item} to {group} is fake and warned people not to share personal details."),
        ("real", "health", "debunk_report", "There is no evidence that {remedy} cures {disease}, doctors at the district hospital in {city} said."),
        ("real", "technology", "debunk_report", "5G towers do not spread {disease}, the telecom department in {state} said, calling the claim baseless."),
        ("real", "economy", "debunk_report", "₹500 notes are not being banned from {day}, the central bank said, asking people to ignore forwarded messages."),
        ("real", "education", "debunk_report", "Board exams for classes 10 and 12 in {state} have not been cancelled, the education board said in a notice."),
        ("real", "govt_schemes", "surprising_true", "The state government will give free bicycles to class 9 girls in {state} under an existing scheme, officials said."),
    ],
    "hinglish": [
        ("real", "transport", "standard", "Railway tyohar ke season mein {n} extra trains chalayega, official release mein bataya gaya."),
        ("real", "weather", "standard", "{city} mein {day} ko bhaari baarish ke kaaran school band rahenge, zila prashasan ne bataya."),
        ("real", "weather", "standard", "Mausam vibhag ke anusaar agle {n_small} din {state} mein baarish ho sakti hai."),
        ("real", "economy", "standard", "Central bank ne {day} ki policy meeting mein repo rate mein koi badlav nahi kiya, bayan jaari hua."),
        ("real", "education", "standard", "{state} board ka class 12 result {month} ke aakhri hafte mein aayega, board ne notice mein kaha."),
        ("real", "health", "standard", "Health department {city} mein bachchon ke liye muft teekakaran abhiyan chala raha hai, adhikariyon ne bataya."),
        ("real", "health", "debunk_report", "Health ministry ne saaf kiya ki {remedy} se {disease} theek hone wala viral message galat hai."),
        ("real", "technology", "debunk_report", "Telecom vibhag ne kaha ki {city} mein 5G towers se {disease} failne ka viral daava jhootha hai."),
        ("real", "elections", "standard", "Election commission ne bataya ki {state} mein {n_small} charan mein voting hogi."),
        ("real", "economy", "surprising_true", "Central bank samvidhan ki varshganth par ₹{coin} ka smarak sikka jaari karega, adhikarik suchna mein kaha gaya."),
        ("real", "transport", "standard", "{city} metro {month} se weekend par der raat tak chalegi, metro nigam ne ghoshna ki."),
        ("real", "govt_schemes", "standard", "Rajya sarkar ne kaha ki is saal {n} lakh {group} ko pension yojana ka laabh mila."),
        ("real", "govt_schemes", "debunk_report", "Adhikariyon ne kaha ki {group} ko free {item} dene wala viral message fake hai, apni jaankari share na karein."),
        ("real", "technology", "surprising_true", "Antariksh agency ne {day} ko mausam satellite safalta se launch kiya, agency ne bayan mein kaha."),
        ("real", "health", "debunk_report", "{remedy} se {disease} theek hone ka koi saboot nahi hai, {city} ke sarkari doctors ne kaha."),
        ("real", "economy", "debunk_report", "{day} se ₹500 ke note band nahi ho rahe, central bank ne forward kiye ja rahe message ko galat bataya."),
        ("real", "education", "debunk_report", "{state} mein class 10 aur 12 ki board pariksha radd nahi hui hai, board ne notice mein kaha."),
    ],
    "hi": [
        ("real", "weather", "standard", "मौसम विभाग के अनुसार इस साल {state} में मानसून सामान्य रहने की संभावना है।"),
        ("real", "weather", "standard", "{city} में {day} को भारी बारिश के कारण स्कूल बंद रहेंगे, जिला प्रशासन ने बताया।"),
        ("real", "transport", "standard", "रेलवे त्योहारों के मौसम में {n} अतिरिक्त ट्रेनें चलाएगा, आधिकारिक विज्ञप्ति में कहा गया।"),
        ("real", "economy", "standard", "केंद्रीय बैंक ने {day} की नीतिगत बैठक में रेपो दर में कोई बदलाव नहीं किया।"),
        ("real", "education", "standard", "{state} बोर्ड कक्षा 12 का परिणाम {month} के अंतिम सप्ताह में जारी करेगा।"),
        ("real", "health", "standard", "स्वास्थ्य विभाग {city} में बच्चों के लिए मुफ्त टीकाकरण अभियान चला रहा है, अधिकारियों ने बताया।"),
        ("real", "health", "debunk_report", "स्वास्थ्य मंत्रालय ने स्पष्ट किया कि {remedy} से {disease} ठीक होने का वायरल संदेश झूठा है।"),
        ("real", "technology", "debunk_report", "दूरसंचार विभाग ने कहा कि {city} में 5G टावरों से {disease} फैलने का वायरल दावा गलत है।"),
        ("real", "elections", "standard", "चुनाव आयोग ने बताया कि {state} में {n_small} चरणों में मतदान होगा।"),
        ("real", "economy", "surprising_true", "केंद्रीय बैंक संविधान की वर्षगांठ पर ₹{coin} का स्मारक सिक्का जारी करेगा, आधिकारिक अधिसूचना में कहा गया।"),
        ("real", "govt_schemes", "standard", "राज्य सरकार ने कहा कि इस साल {n} लाख {group} को पेंशन योजना का लाभ मिला।"),
        ("real", "technology", "surprising_true", "अंतरिक्ष एजेंसी ने {day} को मौसम उपग्रह सफलतापूर्वक कक्षा में स्थापित किया।"),
        ("real", "transport", "standard", "{city} मेट्रो {month} से सप्ताहांत पर देर रात तक चलेगी, मेट्रो निगम ने घोषणा की।"),
        ("real", "health", "debunk_report", "{remedy} से {disease} ठीक होने का कोई प्रमाण नहीं है, {city} के सरकारी डॉक्टरों ने कहा।"),
        ("real", "education", "debunk_report", "{state} में कक्षा 10 और 12 की बोर्ड परीक्षाएं रद्द नहीं हुई हैं, बोर्ड ने सूचना में कहा।"),
        ("real", "economy", "debunk_report", "{day} से 500 के नोट बंद होने का दावा गलत है, केंद्रीय बैंक ने कहा।"),
    ],
}
FAKE_TEMPLATES = {
    "en": [
        ("fake", "health", "standard", "{remedy} cures {disease} in just {n_small} days, doctors are hiding this from you!"),
        ("fake", "health", "standard", "Drinking {remedy} every morning removes {disease} permanently, pharma companies do not want you to know."),
        ("fake", "govt_schemes", "standard", "Government is giving free {item} to all {group}, register at this link before {day}!"),
        ("fake", "govt_schemes", "standard", "New scheme: {group} will get {amount} in their bank account, just fill this form today."),
        ("fake", "economy", "standard", "From {day} ₹500 notes will be banned, exchange them at your bank immediately."),
        ("fake", "economy", "standard", "The central bank is withdrawing ₹{coin} notes from {month}, share this message with your family."),
        ("fake", "technology", "standard", "5G towers are spreading {disease} in {city}, people living near towers are falling sick."),
        ("fake", "technology", "standard", "WhatsApp will start charging ₹{small_amt} per message from {day}, forward this to 10 groups to keep it free."),
        ("fake", "weather", "standard", "NASA has warned that a massive earthquake will hit {state} on {day} night, stay outside your homes."),
        ("fake", "weather", "standard", "A foreign weather agency says a super cyclone will destroy {city} in the next {n_small} days, the government is hiding it."),
        ("fake", "education", "calm_fake", "According to an official notice, board exams for classes 10 and 12 in {state} have been cancelled this year."),
        ("fake", "education", "standard", "All exams in {state} universities are cancelled till {month}, students will be promoted without exams."),
        ("fake", "elections", "standard", "Video of voting machines being hacked in {city} yesterday, the election is rigged!"),
        ("fake", "transport", "standard", "Shocking video: a bridge collapsed in {city} today, {n} feared dead."),
        ("fake", "technology", "standard", "Your SIM card will be blocked within 24 hours, share the OTP sent to your phone to verify your account."),
        ("fake", "technology", "standard", "The telecom department is updating KYC in {state}, reply with your OTP to avoid your number being disconnected."),
        ("fake", "govt_schemes", "calm_fake", "The state government has announced that all {group} will receive {amount} per month from {month}, as per a circular."),
        ("fake", "transport", "calm_fake", "The railways have decided to make train travel free for {group} from {month}, an official circular states."),
        ("fake", "health", "calm_fake", "The district health department in {city} said {remedy} taken daily prevents {disease}, officials said."),
        ("fake", "economy", "calm_fake", "The central bank will stop accepting ₹{coin} coins from {month}, according to an official notification."),
    ],
    "hinglish": [
        ("fake", "govt_schemes", "standard", "Sarkar sabhi {group} ko free {item} de rahi hai, is link par jaldi register karo!"),
        ("fake", "health", "standard", "{remedy} se sirf {n_small} din mein {disease} jad se khatam, doctor ye baat chupa rahe hain!"),
        ("fake", "economy", "standard", "{day} se ₹500 ke note band ho jayenge, turant bank jaakar badal lo."),
        ("fake", "technology", "standard", "WhatsApp {day} se har message ke ₹{small_amt} lega, ise 10 groups mein forward karo."),
        ("fake", "technology", "standard", "5G tower ki wajah se {city} mein {disease} fail raha hai, tower ke paas mat raho."),
        ("fake", "weather", "standard", "NASA ne chetavni di hai ki {day} raat {state} mein bhayankar bhukamp aayega, ghar se bahar raho."),
        ("fake", "education", "calm_fake", "Official notice ke anusaar {state} mein class 10 aur 12 ki board pariksha is saal radd kar di gayi hai."),
        ("fake", "education", "standard", "{state} ki sabhi university exams {month} tak cancel, sabhi students bina exam pass honge."),
        ("fake", "elections", "standard", "{city} mein kal EVM hack hote hue video, chunav mein dhaandhli ho rahi hai!"),
        ("fake", "technology", "standard", "Aapka SIM 24 ghante mein block ho jayega, verify karne ke liye OTP share karein, {state} circle."),
        ("fake", "govt_schemes", "calm_fake", "Rajya sarkar ne ghoshna ki hai ki {month} se sabhi {group} ko har mahine {amount} milenge, circular jaari."),
        ("fake", "transport", "standard", "Abhi abhi {city} mein pul gir gaya, {n} log maare gaye, video dekho."),
        ("fake", "govt_schemes", "standard", "Nayi yojana mein {group} ke khate mein {amount} aayenge, aaj hi ye form bharo."),
        ("fake", "transport", "calm_fake", "Railway ne {month} se {group} ke liye train yatra muft karne ka faisla liya hai, circular mein bataya gaya."),
        ("fake", "economy", "calm_fake", "Central bank {month} se ₹{coin} ke sikke lena band karega, adhikarik suchna mein kaha gaya."),
    ],
    "hi": [
        ("fake", "economy", "standard", "{day} से 500 के नोट बंद हो जाएंगे, यह संदेश सभी को भेजें।"),
        ("fake", "health", "standard", "{remedy} से सिर्फ {n_small} दिन में {disease} खत्म, डॉक्टर यह बात छिपा रहे हैं!"),
        ("fake", "govt_schemes", "standard", "सरकार सभी {group} को मुफ्त {item} दे रही है, इस लिंक पर जल्दी रजिस्टर करें!"),
        ("fake", "technology", "standard", "5G टावर की वजह से {city} में {disease} फैल रहा है, टावर के पास न रहें।"),
        ("fake", "technology", "standard", "व्हाट्सएप {day} से हर संदेश के ₹{small_amt} लेगा, इसे 10 ग्रुप में भेजें।"),
        ("fake", "weather", "standard", "नासा ने चेतावनी दी है कि {day} रात {state} में भयंकर भूकंप आएगा।"),
        ("fake", "education", "calm_fake", "आधिकारिक सूचना के अनुसार {state} में कक्षा 10 और 12 की बोर्ड परीक्षाएं इस साल रद्द कर दी गई हैं।"),
        ("fake", "elections", "standard", "{city} में कल ईवीएम हैक होते हुए वीडियो, चुनाव में धांधली हो रही है!"),
        ("fake", "technology", "standard", "आपका सिम 24 घंटे में बंद हो जाएगा, सत्यापन के लिए ओटीपी साझा करें, {state} सर्कल।"),
        ("fake", "govt_schemes", "calm_fake", "राज्य सरकार ने घोषणा की है कि {month} से सभी {group} को हर महीने {amount} मिलेंगे।"),
        ("fake", "transport", "standard", "अभी-अभी {city} में पुल गिर गया, {n} लोगों की मौत, वीडियो देखें।"),
        ("fake", "education", "standard", "{state} के सभी विश्वविद्यालयों की परीक्षाएं {month} तक रद्द, सभी छात्र बिना परीक्षा पास होंगे।"),
        ("fake", "transport", "calm_fake", "रेलवे ने {month} से {group} के लिए ट्रेन यात्रा मुफ्त करने का निर्णय लिया है।"),
        ("fake", "economy", "calm_fake", "केंद्रीय बैंक {month} से ₹{coin} के सिक्के लेना बंद करेगा, आधिकारिक अधिसूचना में कहा गया।"),
    ],
}

SENSATIONAL = {
    "en": {"prefix": ["BREAKING: ", "🚨 ", "URGENT: ", "🚨🚨 BREAKING: ", "ALERT: "],
           "suffix": [" !!!", " Forward to all groups!", " Share before it is deleted!!", " 😱😱", " Must watch!!!"]},
    "hinglish": {"prefix": ["BREAKING: ", "🚨 ", "URGENT: ", "Bada khulasa: ", "ALERT: "],
                 "suffix": [" !!!", " Sabko bhejo!", " Jaldi share karo!!", " 😱😱", " Forward karo!!!"]},
    "hi": {"prefix": ["ब्रेकिंग: ", "🚨 ", "बड़ी खबर: ", "🚨🚨 ", "सावधान: "],
           "suffix": [" !!!", " सभी को भेजें!", " जल्दी शेयर करें!!", " 😱😱", " ज़रूर देखें!!!"]},
}


def all_templates() -> list[dict]:
    rows = []
    for lang in ("en", "hinglish", "hi"):
        for i, (label, topic, case, text) in enumerate(TEMPLATES[lang] + FAKE_TEMPLATES[lang]):
            rows.append({"template_id": f"{lang}_{label}_{i:02d}", "language": lang, "label": label,
                         "topic": topic, "case_type": case, "template": text})
    return rows


def _fill(template: str, lang: str, rng: random.Random) -> str:
    out = template
    for key, fn in NUMERIC_SLOTS.items():
        while "{" + key + "}" in out:
            out = out.replace("{" + key + "}", fn(rng), 1)
    for key, values in SLOTS[lang].items():
        while "{" + key + "}" in out:
            out = out.replace("{" + key + "}", rng.choice(values), 1)
    if "{" in out:
        raise RuntimeError(f"Unfilled slot in template: {template}")
    return out[0].upper() + out[1:] if lang != "hi" else out


def _sensationalise(text: str, lang: str, rng: random.Random) -> str:
    s = SENSATIONAL[lang]
    text = rng.choice(s["prefix"]) + text
    if rng.random() < 0.8:
        text = text.rstrip(".।") + rng.choice(s["suffix"])
    if lang != "hi" and rng.random() < 0.6:  # one random word in capitals
        words = text.split()
        cand = [i for i, w in enumerate(words) if w.isalpha() and len(w) > 3]
        if cand:
            i = rng.choice(cand)
            words[i] = words[i].upper()
            text = " ".join(words)
    return text


def generate(n_rows: int = 3000, seed: int = 42, style_correlation: float = 0.7, label_noise: float = 0.03,
             language_mix: dict | None = None, n_sources: int = 20, fake_heavy_sources: int = 4,
             start_date: str = "2025-01-01", end_date: str = "2026-10-31") -> pd.DataFrame:
    if n_rows < 50:
        raise FakeNewsError("n_rows must be at least 50.")
    if not 0.0 <= style_correlation <= 1.0:
        raise FakeNewsError("style_correlation must be between 0 and 1.")
    if not 0.0 <= label_noise < 0.5:
        raise FakeNewsError("label_noise must be between 0 and 0.5.")
    language_mix = language_mix or {"en": 0.6, "hinglish": 0.25, "hi": 0.15}
    tot = sum(language_mix.values())
    langs = list(language_mix)
    weights = [language_mix[k] / tot for k in langs]

    rng = random.Random(seed)
    templates = {}
    for t in all_templates():
        templates.setdefault((t["language"], t["label"]), []).append(t)

    d0, d1 = date.fromisoformat(start_date), date.fromisoformat(end_date)
    span = (d1 - d0).days
    sources = [f"channel_{i:02d}" for i in range(1, n_sources + 1)]
    heavy, light = sources[:fake_heavy_sources], sources[fake_heavy_sources:]

    rows, seen = [], set()
    attempts, max_attempts = 0, n_rows * 200
    while len(rows) < n_rows:
        attempts += 1
        if attempts > max_attempts:
            raise FakeNewsError(f"Could only make {len(rows)} unique rows; lower n_rows or add templates.")
        true_label = "fake" if len(rows) % 2 == 0 else "real"  # exact 50/50 before noise
        lang = rng.choices(langs, weights)[0]
        tpl = rng.choice(templates[(lang, true_label)])
        text = _fill(tpl["template"], lang, rng)
        p_sens = style_correlation if true_label == "fake" else 1 - style_correlation
        style = "sensational" if rng.random() < p_sens else "neutral"
        if style == "sensational":
            text = _sensationalise(text, lang, rng)
        if text in seen:
            continue
        seen.add(text)
        if true_label == "fake":
            source = rng.choice(heavy) if rng.random() < 0.5 else rng.choice(sources)
        else:
            source = rng.choice(heavy) if rng.random() < 0.05 else rng.choice(light)
        noisy = rng.random() < label_noise
        label = ({"fake": "real", "real": "fake"}[true_label]) if noisy else true_label
        rows.append({
            "text": text, "label": label, "label_id": 1 if label == "fake" else 0,
            "language": lang, "topic": tpl["topic"], "style": style, "source": source,
            "date": (d0 + timedelta(days=rng.randint(0, span))).isoformat(),
            "case_type": tpl["case_type"], "template_id": tpl["template_id"],
            "true_label": true_label, "noisy_label": noisy,
        })
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))
    return df


def make_dataset(cfg: dict, n_rows=None, seed=None, style_correlation=None, label_noise=None,
                 language_mix=None, out=None) -> Path:
    s = cfg["synthetic"]
    params = dict(
        n_rows=int(n_rows or s["n_rows"]), seed=int(seed if seed is not None else cfg["seed"]),
        style_correlation=float(style_correlation if style_correlation is not None else s["style_correlation"]),
        label_noise=float(label_noise if label_noise is not None else s["label_noise"]),
        language_mix=language_mix or s["language_mix"], n_sources=int(s["n_sources"]),
        fake_heavy_sources=int(s["fake_heavy_sources"]), start_date=s["start_date"], end_date=s["end_date"],
    )
    log(f"Generating {params['n_rows']} synthetic rows (seed={params['seed']}, "
        f"style_correlation={params['style_correlation']}, label_noise={params['label_noise']})")
    df = generate(**params)
    out = Path(out) if out else Path(cfg["paths"]["data_dir"]) / "synthetic" / "synthetic_news.csv"
    ensure_dir(out.parent)
    df.to_csv(out, index=False, encoding="utf-8")
    card = {
        "WARNING": "SYNTHETIC DATA. Generated from templates for building and demonstrating the pipeline. "
                   "Scores on it are NOT model performance. Never publish these items as real posts.",
        "file": str(out), "sha256": file_sha256(out), "params": params, "n_rows": len(df),
        "label_counts": df["label"].value_counts().to_dict(),
        "share_fake": round(float(df["label_id"].mean()), 4),
        "language_counts": df["language"].value_counts().to_dict(),
        "style_by_label": df.groupby("label")["style"].value_counts().unstack(fill_value=0).to_dict(orient="index"),
        "case_type_counts": df["case_type"].value_counts().to_dict(),
        "noisy_labels": int(df["noisy_label"].sum()),
        "n_templates": len(all_templates()),
        "columns": {
            "id": "row number", "text": "the news item", "label / label_id": "fake=1, real=0 (after label noise)",
            "language": "en | hinglish | hi", "topic": "one of 8 topics", "style": "neutral | sensational",
            "source": "synthetic channel name", "date": "random date",
            "case_type": "standard | surprising_true | debunk_report (hard real) | calm_fake (hard fake)",
            "template_id": "which template produced the row (used by the template-held-out audit)",
            "true_label": "label before noise", "noisy_label": "True if the label was flipped on purpose",
        },
    }
    write_json(out.with_suffix(".card.json"), card)
    log(f"Saved {out} ({len(df)} rows, {card['share_fake']:.1%} fake, {card['noisy_labels']} noisy labels)")
    return out
