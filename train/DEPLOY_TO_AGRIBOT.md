# Deploying AgriBanglaT5 into AgriBot

Swaps AgriBot's stock BanglaT5 EN→BN for your adapted model (LoRA adapter +
glossary hints). Same 247M size and CPU latency. **License note:** the adapter is
CC BY-NC-SA 4.0 (from BanglaT5) — fine for research/demo, not commercial.

Edits only `agribot/translation/bangla_t5.py`. Verify before relying on it; the code
falls back to stock BanglaT5 automatically if the adapter is missing.

## 1. Copy the two artifacts into AgriBot
```bash
# the trained adapter
cp -r "f:/Projects/Bangla t5 tune/results/agribanglat5/outputs/agribanglat5-lora" \
      "f:/Projects/Agri_bot/models/agribanglat5-lora"
# the glossary
cp "f:/Projects/Bangla t5 tune/data/glossary.json" \
   "f:/Projects/Agri_bot/data/agri_glossary.json"
```
Then in the AgriBot env: `pip install peft`.

## 2. Three edits to `agribot/translation/bangla_t5.py`

**(a)** In `__init__`, after the EN→BN model is loaded (`self.en_bn_model.eval()`),
add adapter loading + glossary:
```python
        # --- AgriBanglaT5: merge LoRA adapter + load glossary hints (optional) ---
        import json, os
        from pathlib import Path
        self._glossary, self._gloss_forms = [], []
        adapter = Path(self.BASE_DIR := Path(__file__).resolve().parents[2]) / "models" / "agribanglat5-lora"
        gloss = Path(self.BASE_DIR) / "data" / "agri_glossary.json"
        if adapter.exists():
            try:
                from peft import PeftModel
                self.en_bn_model = PeftModel.from_pretrained(self.en_bn_model, str(adapter)).merge_and_unload().eval()
                logger.info("AgriBanglaT5 adapter merged from %s", adapter)
            except Exception as e:
                logger.warning("Adapter load failed (%s); using stock BanglaT5", e)
        if gloss.exists():
            self._glossary = json.loads(gloss.read_text(encoding="utf-8"))
            # (surface_form, bn) longest-first for annotation
            forms = [(f, g["bn"]) for g in self._glossary for f in [g["en"], *g.get("en_aliases", [])] if f]
            self._gloss_forms = sorted(forms, key=lambda x: len(x[0]), reverse=True)
```

**(b)** Add a helper method to the class (anywhere, e.g. after `_normalize_bn`):
```python
    def _annotate_terms(self, text: str) -> str:
        """Append the canonical Bengali term after each glossary English term so
        the adapted model emits the correct terminology (the +5.4pp TermAcc gain)."""
        out = text
        for surface, bn in getattr(self, "_gloss_forms", []):
            if bn in out:
                continue
            pat = re.compile(rf"(?<![A-Za-z])({re.escape(surface)})(?![A-Za-z])", re.IGNORECASE)
            if pat.search(out):
                out = pat.sub(rf"\1 ⟦{bn}⟧", out, count=1)
        return out
```

**(c)** In `_translate_block`, annotate each sentence before tokenizing, and strip
the hint brackets from the output. Change the loop body so the sentence is
annotated and the decoded output has brackets removed:
```python
            ann = self._annotate_terms(sentence)          # <-- add
            inputs = self.en_bn_tokenizer(
                ann,                                       # <-- was: sentence
                return_tensors="pt", max_length=512, truncation=True, padding=True,
            ).to(self.device)
            ...
            output = self.en_bn_tokenizer.decode(generated[0], skip_special_tokens=True)
            output = output.replace("⟦", "").replace("⟧", "")   # <-- strip leftover hint brackets
            translated_parts.append(self._normalize_bn(output))
```

## 3. Verify (before trusting it)
```python
from agribot.translation.bangla_t5 import get_translator
t = get_translator()
print(t.translate_en_to_bn("To control rice blast, apply Tricyclazole at 0.6 g per liter. Do not drain the field."))
# expect: ধানের ব্লাস্ট রোগ ... ০.৬ ... ট্রাইসাইক্লাজল ... নিষ্কাশন ... না  (not "explosion"/"irrigate")
```
Spot-check ~10 real queries against the old output. If anything regresses, just
remove `models/agribanglat5-lora` — the code reverts to stock automatically.
