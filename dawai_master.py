"""
Dawai Master - RapidOCR + RapidFuzz Pipeline
=================================================
Full pipeline: Image -> RapidOCR (EN+AR parallel) -> Top Words by BBox -> rapidfuzz Matcher -> Result
"""
import sys, os, time, re, cv2
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from rapidocr_onnxruntime import RapidOCR
from rapidfuzz import fuzz as rfuzz

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ──────────────────── OCR CORRECTIONS ────────────────────
OCR_CORRECTIONS = {
    "19": 1.0, "1q": 1.0, "I9": 1.0,
    "G25": 625.0, "825": 625.0, "O25": 625.0, "025": 625.0,
    "37S": 375.0, "375": 375.0, "1S6": 156.0, "15G": 156.0
}

ARABIC_FORMS = {
    'أقراص': 'tablets', 'اقراص': 'tablets', 'قرص': 'tablet',
    'شراب': 'syrup', 'معلق': 'suspension',
    'كبسول': 'capsules', 'كبسولات': 'capsules',
    'لبوس': 'suppositories', 'حقن': 'ampoules',
    'جل': 'gel', 'كريم': 'cream', 'مرهم': 'ointment',
    'بخاخ': 'spray', 'قطرة': 'drops', 'قطره': 'drops', 'قطور': 'drops',
    'فوار': 'effervescent', 'اكياس': 'sachets', 'أكياس': 'sachets',
    'غسول': 'lotion', 'محلول': 'solution'
}

# ──────────────────── RTL FIX ────────────────────
def is_arabic(text):
    for ch in text:
        if '\u0600' <= ch <= '\u06FF' or '\u0750' <= ch <= '\u077F' or '\uFB50' <= ch <= '\uFDFF' or '\uFE70' <= ch <= '\uFEFF':
            return True
    return False

def fix_arabic(text):
    if is_arabic(text):
        return text[::-1]
    return text

# ──────────────────── MATCHER (rapidfuzz) ────────────────────
class RapidTwoStageMatcher:
    def __init__(self, db_path='egyptian_medicines.csv'):
        self.db = pd.read_csv(db_path)
        self.parsed_db = []
        self.forms = ['tablets', 'tablet', 'syrup', 'suspension', 'capsules', 'capsule',
                     'suppositories', 'ampoules', 'gel', 'cream', 'ointment', 'spray', 'drops',
                     'injection', 'vials', 'vial', 'sachets', 'sachet', 'inhaler', 'nebules',
                     'effervescent', 'pen', 'flexpen', 'turbuhaler', 'breezhaler', 'ellipta',
                     'solostar', 'flextouch', 'pack', 'enema', 'lotion', 'solution',
                     'im', 'iv', 'sr', 'xr', 'mr', 'cr', 'chewable', 'sublingual',
                     'eye', 'nose', 'ear', 'oral', 'film', 'films']
        self._parse_database()

    def _parse_database(self):
        for idx, row in self.db.iterrows():
            name_en = str(row['Name_EN']).lower()
            strengths = []

            matches = re.finditer(r'(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|iu|%|µg|mmol|million)', name_en)
            for match in matches:
                val = float(match.group(1))
                unit = match.group(2)
                if unit == 'g': val *= 1000.0
                elif unit == 'million': val *= 1000000.0
                strengths.append(val)
                name_en = name_en.replace(match.group(0), '')

            isolated_nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', name_en)
            for n in isolated_nums:
                strengths.append(float(n))
                name_en = re.sub(rf'\b{n}\b', '', name_en)

            found_form = None
            for form in self.forms:
                if re.search(rf'\b{form}\b', name_en):
                    if not found_form:
                        found_form = form
                    name_en = re.sub(rf'\b{form}\b', '', name_en)

            clean_en = name_en.lower()
            for f in self.forms: clean_en = re.sub(rf'\b{f}\b', '', clean_en)
            clean_en = re.sub(r'\b\d+(?:\.\d+)?\s*(mg|ml|mcg|iu|g|%|µg|mmol)\b', '', clean_en)
            clean_en = re.sub(r'[^a-zA-Z0-9\s]', ' ', clean_en)
            brand_core = ' '.join(clean_en.split()).strip()

            self.parsed_db.append({
                'original': str(row['Name_EN']),
                'brand_core': brand_core,
                'strengths': list(set(strengths)),
                'form': found_form,
                'raw_row': row
            })
        print(f"📊 Matcher: {len(self.parsed_db)} medicines loaded.", flush=True)

    def find_family(self, ocr_chunks):
        suspected_brands = []
        for text, height, bbox in ocr_chunks:
            if len(text) < 3: continue
            if not any(c.isalpha() for c in text): continue
            if bool(re.search(r'[@*#%]', text)): continue
            valid_words = [w for w in re.findall(r'[a-zA-Z\u0600-\u06FF]+', text) if len(w) > 2]
            if not valid_words: continue

            text_lower = text.lower()
            for chem in ['hydrochloride', 'sodium', 'potassium', 'sulfate', 'maleate']:
                text_lower = text_lower.replace(chem, ' ')

            for candidate in self.parsed_db:
                clean_text = re.sub(r'\b\d+(?:\.\d+)?\s*(mg|ml|mcg|iu|g|%|µg|mmol|مجم|ملجم|مل|جم|وحدة)\b', '', text_lower, flags=re.IGNORECASE)
                clean_text = re.sub(r'[/%\-\(\)\[\]]', ' ', clean_text).strip()
                clean_candidate = candidate['brand_core'].lower()

                # ⚡ rapidfuzz instead of thefuzz
                score = rfuzz.token_set_ratio(clean_candidate, clean_text)
                if len(clean_text) > 4:
                    p_score_en = rfuzz.partial_ratio(clean_candidate, clean_text)
                    score = max(score, p_score_en)

                ar_name = str(candidate['raw_row'].get('Name_AR', ''))
                if ar_name and ar_name != 'nan':
                    ar_forms = r'\b(أقراص|اقراص|كبسولات|شراب|معلق|لبوس|امبولات|امبول|فوار|قطرة|قطره|مرهم|كريم|جل|حقن|حقنة|حقنه|فيال|نقط)\b'
                    
                    clean_ar_candidate = re.sub(r'\d+(?:\.\d+)?\s*(مجم|ملجم|مل|جم|وحدة|mg|ml|%)', '', ar_name)
                    clean_ar_candidate = re.sub(ar_forms, '', clean_ar_candidate)
                    clean_ar_candidate = re.sub(r'[أإآ]', 'ا', clean_ar_candidate)
                    clean_ar_candidate = re.sub(r'ة', 'ه', clean_ar_candidate)
                    clean_ar_candidate = re.sub(r'ى', 'ي', clean_ar_candidate)
                    clean_ar_candidate = clean_ar_candidate.strip()

                    clean_ar_text = re.sub(r'\d+(?:\.\d+)?\s*(مجم|ملجم|مل|جم|وحدة|mg|ml|%)', '', text)
                    clean_ar_text = re.sub(ar_forms, '', clean_ar_text)
                    clean_ar_text = re.sub(r'[أإآ]', 'ا', clean_ar_text)
                    clean_ar_text = re.sub(r'ة', 'ه', clean_ar_text)
                    clean_ar_text = re.sub(r'ى', 'ي', clean_ar_text)
                    clean_ar_text = clean_ar_text.strip()
                    
                    ar_score = rfuzz.token_set_ratio(clean_ar_candidate, clean_ar_text)
                    if len(clean_ar_text) > 4:
                        p_score = rfuzz.partial_ratio(clean_ar_candidate, clean_ar_text)
                        ar_score = max(ar_score, p_score)
                    score = max(score, ar_score)

                if score >= 90:
                    suspected_brands.append((score, candidate['brand_core']))

        if suspected_brands:
            suspected_brands.sort(key=lambda x: x[0], reverse=True)
            unique_brands_scores = {}
            for sc, br in suspected_brands:
                if br not in unique_brands_scores:
                    unique_brands_scores[br] = sc

            best_score = suspected_brands[0][0]
            if best_score >= 90:
                best_br = suspected_brands[0][1]
                unique_brands_scores = {best_br: best_score}
            else:
                unique_brands_scores = dict(list(unique_brands_scores.items())[:3])

            subset = []
            for c in self.parsed_db:
                br = c['brand_core']
                if br in unique_brands_scores:
                    c_copy = c.copy()
                    c_copy['family_score'] = unique_brands_scores[br]
                    subset.append(c_copy)

            return subset, None, best_score
        return None, None, 0

    def resolve_variant(self, subset, raw_text):
        best_match = subset[0]
        highest_score = -9999
        text = raw_text.lower()

        extracted_numbers = []
        unit_matches = re.finditer(r'(\d+(?:\.\d+)?)\s*(mg|ml|mcg|iu|g|%|µg|mmol|مجم|ملجم|مل|جم|وحدة)', text)
        for match in unit_matches:
            val = float(match.group(1))
            unit = match.group(2)
            if unit in ('g', 'جم'): val *= 1000.0
            extracted_numbers.append(val)

        for error_k, correct_v in OCR_CORRECTIONS.items():
            if error_k.lower() in text:
                extracted_numbers.append(correct_v)

        extracted_forms = [f for f in self.forms if f in text]
        for ar_form, en_form in ARABIC_FORMS.items():
            if ar_form in text:
                extracted_forms.append(en_form)
            else:
                for word in text.split():
                    if len(word) > 3 and rfuzz.ratio(ar_form, word) >= 75:
                        extracted_forms.append(en_form)
                        break
        extracted_forms = list(set(extracted_forms))

        for candidate in subset:
            family_score = candidate.get('family_score', 100)
            family_w = family_score / 100.0
            evidence = 0

            if candidate['strengths'] and extracted_numbers:
                c_strength = candidate['strengths'][0]
                if any(abs(c_strength - n) < 0.1 or abs((c_strength / 1000.0) - n) < 0.1 for n in extracted_numbers):
                    evidence += 50

            if candidate['form']:
                if candidate['form'] in extracted_forms:
                    evidence += 30

            variant_words = [w for w in candidate['original'].lower().split()
                           if len(w) >= 3 and w not in ['tablets', 'syrup', 'capsules', 'mg', 'ml', 'g',
                                                          candidate['brand_core'].split()[0]]]
            for vw in variant_words:
                if rfuzz.partial_ratio(vw, text) >= 75:
                    evidence += 60

            final_score = family_score + (evidence * family_w)
            if final_score > highest_score:
                highest_score = final_score
                best_match = candidate

        return best_match, highest_score

# ──────────────────── OCR ENGINE ────────────────────
_rapid_en = None
_rapid_ar = None
_matcher = None

def get_ocr_engines():
    global _rapid_en, _rapid_ar
    if _rapid_en is None:
        print("🔧 Loading RapidOCR English model...", flush=True)
        _rapid_en = RapidOCR()
        ar_model = os.path.join(os.path.dirname(__file__), 'models', 'arabic_ocr', 'languages', 'arabic', 'rec.onnx')
        ar_dict = os.path.join(os.path.dirname(__file__), 'models', 'arabic_ocr', 'languages', 'arabic', 'dict.txt')
        print("🔧 Loading RapidOCR Arabic model...", flush=True)
        _rapid_ar = RapidOCR(rec_model_path=ar_model, rec_keys_path=ar_dict)
    return _rapid_en, _rapid_ar

def get_matcher():
    global _matcher
    if _matcher is None:
        db_path = os.path.join(os.path.dirname(__file__), 'egyptian_medicines.csv')
        _matcher = RapidTwoStageMatcher(db_path)
    return _matcher

def process_image(image_path):
    """Full pipeline: Image -> RapidOCR -> Matcher -> Result"""
    total_start = time.time()

    # 1. Load image + fix EXIF orientation + preprocess
    from PIL import Image, ImageOps
    import numpy as np
    
    try:
        pil_img = Image.open(image_path)
        # تعديل الصورة لو كانت مقلوبة بناءً على الـ EXIF
        pil_img = ImageOps.exif_transpose(pil_img)
        # تحويل الصورة من PIL (RGB) إلى OpenCV (BGR)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"⚠️ Error loading image: {e}")
        img = None
        
    if img is None:
        return {"found": False, "error": "Image not found"}

    max_dim = 1200
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 2. Run OCR (EN + AR in parallel)
    rapid_en, rapid_ar = get_ocr_engines()

    t_ocr_start = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        # RapidOCR performs exceptionally well with raw BGR images.
        f_en = pool.submit(lambda: rapid_en(img))
        f_ar = pool.submit(lambda: rapid_ar(img))
        result_en, _ = f_en.result()
        result_ar, _ = f_ar.result()
    t_ocr = time.time() - t_ocr_start

    # 3. Merge results + fix Arabic RTL
    all_ocr = []
    if result_en:
        for line in result_en:
            box, text, conf = line
            ys = [p[1] for p in box]
            box_h = max(ys) - min(ys)
            all_ocr.append((text, box_h, box, conf, 'EN'))

    if result_ar:
        for line in result_ar:
            box, text, conf = line
            text = fix_arabic(text)
            ys = [p[1] for p in box]
            box_h = max(ys) - min(ys)
            all_ocr.append((text, box_h, box, conf, 'AR'))

    if not all_ocr:
        return {"found": False, "error": "No text detected"}

    # 4. Pick top 3 biggest words (by bounding box height) for family search
    all_ocr.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [(text, h, box) for text, h, box, conf, lang in all_ocr[:3]]
    raw_text = ' '.join([t[0] for t in all_ocr])

    # 5. Match against database
    matcher = get_matcher()

    t_match_start = time.time()
    subset, _, family_score = matcher.find_family(top_chunks)

    if subset:
        best_match, variant_score = matcher.resolve_variant(subset, raw_text)
        t_match = time.time() - t_match_start
        row = best_match['raw_row']
        total_time = time.time() - total_start
        print(f"[API] Processed Image -> FOUND: {row['Name_EN']} (Time: {total_time:.2f}s)", flush=True)

        return {
            "found": True,
            "medicine_id": int(row.get('id', 0)),
            "Name_EN": str(row['Name_EN']),
            "Name_AR": str(row.get('Name_AR', '')),
            "Category": str(row.get('Category', '')),
            "Uses": str(row.get('Uses', '')),
            "SideEffects": str(row.get('SideEffects', '')),
            "confidence": family_score,
            "time_ocr": round(t_ocr, 2),
            "time_match": round(t_match, 4),
            "time_total": round(total_time, 2)
        }
    else:
        total_time = time.time() - total_start
        print(f"[API] Processed Image -> NOT FOUND (Time: {total_time:.2f}s)", flush=True)
        return {"found": False, "time_total": round(total_time, 2)}


