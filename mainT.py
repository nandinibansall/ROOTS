
import fitz 
from deep_translator import GoogleTranslator

WHITE = fitz.utils.getColor("white")

textflags = fitz.TEXT_DEHYPHENATE

to_hindi = GoogleTranslator(source='en', target='hi')

doc = fitz.open('claimformat.pdf')

ocg = doc.add_ocg("Hindi", on=True)

for page in doc:
    blocks = page.get_text("blocks", flags=textflags)
    for block in blocks:
        bbox = block[:4] 
        text = block[4]  
        hindi = to_hindi.translate(text) 
        page.draw_rect(bbox, color=None, fill=WHITE, oc=ocg)
        page.insert_htmlbox(
            bbox,
            hindi,
            css="* {font-family: sans-serif;}", 
            oc=ocg
        )

doc.subset_fonts()

doc.ez_save('translated.pdf')
