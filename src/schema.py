import random

ID_TO_LABEL = {
    0: 'Blank',
    1: 'Text',
    2: 'TextInlineMath',
    3: 'Code',
    4: 'SectionHeader',
    5: 'Caption',
    6: 'Footnote',
    7: 'Equation',
    8: 'ListItem',
    9: 'PageFooter',
    10: 'PageHeader',
    11: 'Picture',
    12: 'Figure',
    13: 'Table',
    14: 'Form',
    15: 'TableOfContents',
    16: 'Handwriting'
}

LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}
LABEL_COUNT = len(ID_TO_LABEL)

COLOR_MAP={}
for key,_ in LABEL_TO_ID.items():
  COLOR_MAP[key]=f"#{''.join([random.choice('0123456789ABCDEF') for j in range(6)])}"


text_labels={'Text','SectionHeader','Footnote','TextInlineMath',
             'Code','Caption','Equation','PageFooter','PageHeader',
             'ListItem','Form'}

pad_size=30
buffer_space=2

non_text_labels=set([key for key in set(COLOR_MAP.keys()) if key not in text_labels])
