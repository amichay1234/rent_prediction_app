import pandas as pd
import numpy as np
import re

def prepare_data(dataset, is_prediction=False):
    # המרה בטוחה של תיאור
    dataset['description'] = dataset['description'].fillna('').astype(str)

    # עיבוד מחיר – רק אם לא במצב חיזוי
    if not is_prediction and 'price' in dataset.columns:
        dataset['price'] = pd.to_numeric(dataset['price'], errors='coerce')
        dataset.dropna(subset=['price'], inplace=True)
        dataset = dataset[(dataset['price'] >= 1500) & (dataset['price'] <= 18500)]

    # המרה של ערכים חסרי מידע
    dataset.replace({'': np.nan, ' ': np.nan}, inplace=True)

    # הסרה אם אין neighborhood וגם אין address
    dataset = dataset[~(dataset["neighborhood"].isna() & dataset["address"].isna())].reset_index(drop=True)

    # סינון לפי שדות לא חסרים (לא לפי 0)
    dataset = dataset[dataset.notna().sum(axis=1) > 4]

    if dataset.empty:
        raise ValueError("לא נשארו שורות אחרי סינון הנתונים. ודא שכל השדות מולאו כראוי.")

    # סיווג טיפוס נכס
    def classify_property_type(val): 
        if not isinstance(val, str):
            return np.nan
        val_lower = val.lower()
        if 'סאבלט' in val_lower:
            return 'סאבלט'
        elif 'פרטי' in val_lower or 'קוטג' in val_lower or 'דו משפחתי' in val_lower:
            return 'פרטי'
        elif 'סטודיו' in val_lower or 'לופט' in val_lower:
            return 'סטודיו/לופט'
        elif 'גג' in val_lower or 'פנטהאוז' in val_lower:
            return 'גג/פנטהאוז'
        elif 'דופלקס' in val_lower:
            return 'דופלקס'
        elif 'דירת גן' in val_lower:
            return 'דירת גן'
        elif 'דירה' in val_lower or 'דירת' in val_lower:
            return 'דירה'
        else:
            return np.nan

    # ניחוש סוג דירה מתוך התיאור
    def guess_type_from_description(desc):
        if pd.isna(desc):
            return np.nan
        desc = desc.lower()
        patterns = [
            (r'\bיחידת דיור\b', 'יחידת דיור'),
            (r'\bסאבלט\b', 'סאבלט'),
            (r'\b(פרטי|קוטג|דו משפחתי)\b', 'פרטי'),
            (r'\b(סטודיו|לופט)\b', 'סטודיו/לופט'),
            (r'\b(גג|פנטהאוז)\b', 'גג/פנטהאוז'),
            (r'\bדופלקס\b', 'דופלקס'),
            (r'\bדירת גן\b', 'דירת גן'),
            (r'\b(דירה|דירת)\b', 'דירה'),
            (r'\bחדר\b', 'דירה'),
            (r'\bבית\b', 'דירה'),
        ]
        for pattern, value in patterns:
            if re.search(pattern, desc):
                return value
        return np.nan

    # השלמה אם property_type חסרה
    if 'property_type' not in dataset.columns:
        dataset['property_type'] = np.nan

    # סיווג טיפוס דירה
    dataset['property_type_clean'] = dataset['property_type'].apply(classify_property_type)

    # ניחוש לפי description
    missing_mask = dataset['property_type_clean'].isna()
    dataset.loc[missing_mask, 'property_type_clean'] = dataset.loc[missing_mask, 'description'].apply(guess_type_from_description)

    # מילוי לא ידוע
    dataset['property_type_clean'] = dataset['property_type_clean'].fillna('לא ידוע')

    # סינון שורות עם "קורות חיים"
    dataset['description'] = dataset['description'].fillna('').astype(str)
    dataset = dataset[~dataset['description'].str.contains('קורות חיים', case=False, na=False)]
    dataset = dataset[dataset['property_type_clean'] != 'לא ידוע']

    if dataset.empty:
        raise ValueError("לא נותרו שורות לאחר סינון לפי סוג דירה או תיאור.")

    # ✅ קטגוריה לפי שטח גינה
    def categorize_garden_area(x):
        if pd.isna(x):
            return 'לא ידוע'
        elif x == 0:
            return 'אין גינה'
        elif x <= 20:
            return 'קטנה'
        elif x <= 50:
            return 'בינונית'
        else:
            return 'גדולה'

    dataset['garden_size_category'] = dataset['garden_area_filled'].apply(categorize_garden_area)
    dataset['garden_size_category'] = dataset['garden_size_category'].astype('category')

    # ✅ קטגוריה לפי דמי ועד בניין
    def categorize_building_tax(value):
        if pd.isna(value):
            return "לא יודע"
        elif 0 <= value < 500:
            return "זול"
        elif 500 <= value < 1000:
            return "ממוצע"
        elif 1000 <= value <= 1500:
            return "גבוה"
        else:
            return "לא ידוע"

    dataset["building_tax_category"] = dataset["building_tax"].apply(categorize_building_tax)
    dataset["building_tax_category"] = dataset["building_tax_category"].astype("category")

    # ✅ חילוץ שם רחוב מתוך כתובת
    def extract_street(address):
        if pd.isna(address):
            return np.nan
        address = str(address)
        match = re.search(r"([^0-9]+)", address)
        if match:
            return match.group(1).strip()
        return np.nan

    dataset["street"] = dataset["address"].apply(extract_street)

    return dataset
